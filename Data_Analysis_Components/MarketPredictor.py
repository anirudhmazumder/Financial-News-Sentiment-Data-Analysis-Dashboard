from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re

class MarketPredictor:
    def __init__(self, sentiment_analysis_path):
        self.sentiment_analysis_path = sentiment_analysis_path
        self.regression_models = {
            'random_forest_regressor': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'linear_regression': LinearRegression()
        }
        self.classification_models = {
            'random_forest_classifier': RandomForestClassifier(),
            'logistic_regression': LogisticRegression()
        }
        self.feature_columns = []
    
    def _parse_time(self, s):
            s = str(s).strip()
            if not s or s.lower().startswith('loading'):
                return pd.NaT
            # Date only: YYYY-MM-DD
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
                return pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
            # Try date + time with AM/PM first
            try:
                return pd.to_datetime(s, format='%Y-%m-%d %I:%M%p', errors='raise')
            except Exception:
                # Fallback to pandas inference
                return pd.to_datetime(s, errors='coerce')
    
    def load_and_preprocess_data(self):
        sentiment_analysis_df = pd.read_csv(self.sentiment_analysis_path)

        sentiment_analysis_df['Date'] = sentiment_analysis_df['Time'].apply(self._parse_time)
        sentiment_analysis_df = sentiment_analysis_df.dropna(subset=['Date'])
        sentiment_analysis_df = sentiment_analysis_df.sort_values('Date').reset_index(drop=True)
        
        return sentiment_analysis_df

    def daily_sentiment_aggregation(self, sentiment_analysis_df):
        sentiment_analysis_df['sentiment_score'] = (
            sentiment_analysis_df['Positive_Probability'] - sentiment_analysis_df['Negative_Probability']
        )
        
        # Weighted sentiment (confidence-weighted)
        sentiment_analysis_df['weighted_sentiment'] = (
            sentiment_analysis_df['sentiment_score'] * sentiment_analysis_df['Confidence']
        )
        
        # Daily aggregations
        daily_data_aggregation = sentiment_analysis_df.groupby('Date').agg({
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
            'weighted_sentiment': ['mean', 'std'],
            'Confidence': ['mean', 'std'],
            'Positive_Probability': ['mean', 'max'],
            'Negative_Probability': ['mean', 'max'],
            'Neutral_Probability': ['mean', 'max']
        }).round(4)
        
        # Flatten column names
        daily_data_aggregation.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                           for col in daily_data_aggregation.columns]

        # Rename for clarity
        daily_data_aggregation = daily_data_aggregation.rename(columns={
            'sentiment_score_count': 'headline_count',
            'sentiment_score_mean': 'avg_sentiment',
            'sentiment_score_std': 'sentiment_volatility',
            'sentiment_score_min': 'min_sentiment',
            'sentiment_score_max': 'max_sentiment',
            'weighted_sentiment_mean': 'avg_weighted_sentiment',
            'weighted_sentiment_std': 'weighted_sentiment_volatility',
            'Confidence_mean': 'avg_confidence',
            'Confidence_std': 'confidence_volatility'
        })
        
        # Handle missing values (days with only 1 headline)
        daily_data_aggregation = daily_data_aggregation.fillna(0)
        
        return daily_data_aggregation.reset_index()

    def create_rolling_features(self, df, windows=[3, 7, 14, 30]):
        df = df.sort_values('Date').reset_index(drop=True)
        
        for window in windows:
            df[f'sentiment_ma_{window}d'] = df['avg_sentiment'].rolling(window, min_periods=1).mean()
            df[f'sentiment_std_{window}d'] = df['avg_sentiment'].rolling(window, min_periods=1).std()
            df[f'weighted_sentiment_ma_{window}d'] = df['avg_weighted_sentiment'].rolling(window, min_periods=1).mean()
            
            # Rolling headline count
            df[f'headline_count_ma_{window}d'] = df['headline_count'].rolling(window, min_periods=1).mean()
            
            # Rolling confidence
            df[f'confidence_ma_{window}d'] = df['avg_confidence'].rolling(window, min_periods=1).mean()
            
            # Sentiment momentum (change from previous period)
            df[f'sentiment_momentum_{window}d'] = (
                df[f'sentiment_ma_{window}d'] - df[f'sentiment_ma_{window}d'].shift(1)
            )
            
            # Sentiment trend (linear regression slope over window)
            df[f'sentiment_trend_{window}d'] = (
                df['avg_sentiment'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            )
        
        return df
    
    def create_sentiment_features(self, df):
        sentiment_q75 = df['avg_sentiment'].quantile(0.75)
        sentiment_q25 = df['avg_sentiment'].quantile(0.25)
        
        df['extreme_positive_day'] = (df['avg_sentiment'] > sentiment_q75).astype(int)
        df['extreme_negative_day'] = (df['avg_sentiment'] < sentiment_q25).astype(int)
        
        # High volatility sentiment days
        volatility_q75 = df['sentiment_volatility'].quantile(0.75)
        df['high_sentiment_volatility'] = (df['sentiment_volatility'] > volatility_q75).astype(int)
        
        # High news flow days
        count_q75 = df['headline_count'].quantile(0.75)
        df['high_news_flow_day'] = (df['headline_count'] > count_q75).astype(int)
        
        # Mixed sentiment days (high neutral probability)
        df['mixed_sentiment_day'] = (df['Neutral_Probability_mean'] > 0.5).astype(int)
        
        # High confidence days
        df['high_confidence_day'] = (df['avg_confidence'] > 0.8).astype(int)
        
        return df

    def create_keyword_based_features(self, sentiment_analysis_df, daily_df):
        if 'sentiment_score' not in sentiment_analysis_df.columns:
            sentiment_analysis_df['sentiment_score'] = (
                sentiment_analysis_df['Positive_Probability'] - sentiment_analysis_df['Negative_Probability']
            )

        keywords = {
            'bullish': ['surge', 'soar', 'rally', 'gain', 'rise', 'up', 'high', 'record', 'breakthrough'],
            'bearish': ['fall', 'drop', 'decline', 'down', 'low', 'crash', 'plunge', 'threat', 'risk'],
            'fed_monetary': ['fed', 'federal reserve', 'interest rate', 'monetary policy', 'powell', 'rate cut', 'rate hike'],
            'earnings': ['earnings', 'revenue', 'profit', 'guidance', 'beat', 'miss', 'forecast'],
            'trade_tariffs': ['tariff', 'trade war', 'trade', 'china', 'trump', 'trade deal'],
            'tech_ai': ['ai', 'artificial intelligence', 'technology', 'semiconductor', 'tech', 'nvidia', 'tesla'],
            'geopolitical': ['war', 'conflict', 'sanctions', 'russia', 'ukraine', 'military', 'threat'],
            'economic': ['gdp', 'inflation', 'unemployment', 'economic', 'recession', 'growth', 'recovery']
        }
        
        # Calculate daily keyword scores
        keyword_features = []
        
        for date in daily_df['Date']:
            day_headlines = sentiment_analysis_df[sentiment_analysis_df['Date'] == date]
            day_features = {'Date': date}
            
            for category, words in keywords.items():
                # Count mentions
                mentions = day_headlines['Headline'].str.lower().apply(
                    lambda x: sum(word in str(x).lower() for word in words) if pd.notna(x) else 0
                ).sum()
                
                # Weighted sentiment for this category
                relevant_headlines = day_headlines[
                    day_headlines['Headline'].str.lower().apply(
                        lambda x: any(word in str(x).lower() for word in words) if pd.notna(x) else False
                    )
                ]
                
                if len(relevant_headlines) > 0:
                    category_sentiment = (
                        relevant_headlines['sentiment_score'] * relevant_headlines['Confidence']
                    ).mean()
                else:
                    category_sentiment = 0
                
                day_features[f'{category}_mentions'] = mentions
                day_features[f'{category}_sentiment'] = category_sentiment
                day_features[f'{category}_headlines'] = len(relevant_headlines)
            
            keyword_features.append(day_features)
        
        keyword_df = pd.DataFrame(keyword_features)
        
        # Merge with daily data
        daily_df = daily_df.merge(keyword_df, on='Date', how='left').fillna(0)
        
        return daily_df
    
    def fetch_market_data(self, start_date, end_date, symbols=['SPY', '^VIX', '^TNX']):
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date + timedelta(days=30))  # Extra buffer
                
                if symbol == 'SPY':
                    # S&P 500 ETF - main market indicator
                    data['daily_return'] = data['Close'].pct_change()
                    data['volatility_5d'] = data['daily_return'].rolling(5).std() * np.sqrt(252)
                    data['sma_20'] = data['Close'].rolling(20).mean()
                    data['rsi'] = self.calculate_rsi(data['Close'])
                    data['market_trend'] = (data['Close'] > data['sma_20']).astype(int)
                    
                elif symbol == '^VIX':
                    # VIX - Fear index
                    data = data.rename(columns={'Close': 'VIX'})
                    data['vix_change'] = data['VIX'].pct_change()
                    data['high_fear'] = (data['VIX'] > 20).astype(int)
                    
                elif symbol == '^TNX':
                    # 10-Year Treasury yield
                    data = data.rename(columns={'Close': 'treasury_yield'})
                    data['yield_change'] = data['treasury_yield'].pct_change()
                
                market_data[symbol] = data
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return market_data
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_target_variables(self, daily_df, market_data):
        if 'SPY' in market_data:
            spy_data = market_data['SPY'][['daily_return', 'volatility_5d', 'market_trend']].reset_index()
            spy_data['Date'] = spy_data['Date'].dt.date
            daily_df['Date_for_merge'] = daily_df['Date'].dt.date
            
            # Merge market data
            merged = daily_df.merge(spy_data, left_on='Date_for_merge', right_on='Date', 
                                  how='left', suffixes=('', '_market'))
            
            # Create forward-looking targets (next day's performance)
            merged['next_day_return'] = merged['daily_return'].shift(-1)
            merged['next_day_positive'] = (merged['next_day_return'] > 0).astype(int)
            merged['next_day_big_move'] = (abs(merged['next_day_return']) > 0.01).astype(int)  # >1% move
            
            # Multi-day targets
            merged['next_3day_return'] = merged['daily_return'].rolling(3).mean().shift(-3)
            merged['next_3day_positive'] = (merged['next_3day_return'] > 0).astype(int)
            
            # Add VIX data if available
            if '^VIX' in market_data:
                vix_data = market_data['^VIX'][['VIX', 'vix_change', 'high_fear']].reset_index()
                vix_data['Date'] = vix_data['Date'].dt.date
                merged = merged.merge(vix_data, left_on='Date_for_merge', right_on='Date', 
                                    how='left', suffixes=('', '_vix'))
            
            return merged.drop(['Date_for_merge', 'Date_market', 'Date_vix'], axis=1, errors='ignore')
        
        else:
            print("No market data available. Creating synthetic targets based on sentiment.")
            # Create synthetic targets based on future sentiment
            daily_df['next_day_sentiment'] = daily_df['avg_sentiment'].shift(-1)
            daily_df['next_day_positive'] = (daily_df['next_day_sentiment'] > 0).astype(int)
            daily_df['next_3day_sentiment'] = daily_df['avg_sentiment'].rolling(3).mean().shift(-3)
            daily_df['next_3day_positive'] = (daily_df['next_3day_sentiment'] > 0).astype(int)
            
            return daily_df
        
    def prepare_features_for_modeling(self, data):
        exclude_cols = [
            'Date', 'next_day_return', 'next_day_positive', 'next_day_big_move',
            'next_3day_return', 'next_3day_positive', 'next_day_sentiment', 'next_3day_sentiment',
            'daily_return', 'Date_y'  # Current day market data shouldn't be used to predict itself
        ]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Unnamed')]
        
        # Handle missing values
        X = data[feature_cols].fillna(method='ffill').fillna(0)
        
        self.feature_columns = feature_cols
        
        return X

    def train_and_evaluate_models(self, X, y, model_type='classification'):
        if len(X) < 5:
            print(f"Warning: Only {len(X)} samples available. Cannot train models.")
            return {}
        elif len(X) < 10:
            print(f"Warning: Only {len(X)} samples available. Using simple train-test split.")
            # For very small datasets, use a simple approach
            train_size = max(3, int(0.7 * len(X)))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            n_splits = 3
            if len(X_test) == 0:
                # If no test data, use training data for evaluation (not ideal but necessary)
                X_test, y_test = X_train, y_train
                print("   Note: Using training data for evaluation due to small dataset size.")
        else:
            n_splits = min(3, len(X) // 5)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = self.classification_models if model_type == 'classification' else self.regression_models
        results = {}
        
        for name, model in models.items():
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                if model_type == 'classification':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = r2_score(y_val, y_pred)
                
                scores.append(score)
            
            # Final fit on all data
            model.fit(X, y)
            
            results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'model': model
            }
        
        return results
    
    def get_feature_importances(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': abs(model.coef_)
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def prediction_details(self, daily_df, results, target_col):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sentiment over time
        axes[0, 0].plot(daily_df['Date'], daily_df['avg_sentiment'], label='Daily Avg Sentiment')
        axes[0, 0].plot(daily_df['Date'], daily_df['sentiment_ma_7d'], label='7-day MA', alpha=0.7)
        axes[0, 0].set_title('Daily Sentiment Evolution')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sentiment distribution
        axes[0, 1].hist(daily_df['avg_sentiment'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(daily_df['avg_sentiment'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_title('Daily Sentiment Distribution')
        axes[0, 1].legend()
        
        # 3. Headlines count over time
        axes[0, 2].bar(daily_df['Date'], daily_df['headline_count'], alpha=0.7)
        axes[0, 2].set_title('Daily Headlines Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Sentiment vs Market Performance (if available)
        if 'next_day_return' in daily_df.columns:
            valid_data = daily_df.dropna(subset=['next_day_return', 'avg_sentiment'])
            axes[1, 0].scatter(valid_data['avg_sentiment'], valid_data['next_day_return'], alpha=0.6)
            axes[1, 0].set_xlabel('Daily Sentiment')
            axes[1, 0].set_ylabel('Next Day Return')
            axes[1, 0].set_title('Sentiment vs Market Performance')
            
            # Add correlation
            corr = valid_data['avg_sentiment'].corr(valid_data['next_day_return'])
            axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 0].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 5. Model performance comparison
        if results:
            model_names = list(results.keys())
            scores = [results[name]['mean_score'] for name in model_names]
            axes[1, 1].bar(model_names, scores)
            axes[1, 1].set_title(f'Model Performance ({target_col})')
            axes[1, 1].set_ylabel('Score')
            
        # 6. Feature importance (top 10)
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_score'])
        best_model = results[best_model_name]['model']
        importance_df = self.get_feature_importance(best_model, self.feature_columns)
        
        if importance_df is not None:
            top_features = importance_df.head(10)
            axes[1, 2].barh(range(len(top_features)), top_features['importance'])
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features['feature'])
            axes[1, 2].set_title('Top 10 Feature Importance')
            axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        plt.savefig('prediction_dashboard.png')
        
        return fig
    
    def create_simplified_dashboard(self, daily_df):
        """Create simplified dashboard for small datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Sentiment over time
        axes[0, 0].plot(daily_df['Date'], daily_df['avg_sentiment'], 'b-o', label='Daily Avg Sentiment')
        if len(daily_df) > 3:
            axes[0, 0].plot(daily_df['Date'], daily_df['sentiment_ma_3d'], 'r--', label='3-day MA', alpha=0.7)
        axes[0, 0].set_title('Daily Sentiment Evolution')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sentiment distribution
        axes[0, 1].hist(daily_df['avg_sentiment'], bins=min(10, len(daily_df)), alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(daily_df['avg_sentiment'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_title('Daily Sentiment Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Headlines count over time
        axes[1, 0].bar(daily_df['Date'], daily_df['headline_count'], alpha=0.7, color='green')
        axes[1, 0].set_title('Daily Headlines Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sentiment vs Market Performance (if available)
        if 'next_day_return' in daily_df.columns and 'daily_return' in daily_df.columns:
            valid_data = daily_df.dropna(subset=['daily_return', 'avg_sentiment'])
            if len(valid_data) > 0:
                axes[1, 1].scatter(valid_data['avg_sentiment'], valid_data['daily_return'], 
                                 alpha=0.8, s=100, c='red')
                axes[1, 1].set_xlabel('Daily Sentiment')
                axes[1, 1].set_ylabel('Market Return')
                axes[1, 1].set_title('Sentiment vs Market Performance')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add correlation if we have enough data points
                if len(valid_data) > 2:
                    corr = valid_data['avg_sentiment'].corr(valid_data['daily_return'])
                    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                                   transform=axes[1, 1].transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='wheat'))
            else:
                axes[1, 1].text(0.5, 0.5, 'No market data\navailable for comparison', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Market Performance')
        else:
            # Show confidence vs sentiment instead
            axes[1, 1].scatter(daily_df['avg_confidence'], daily_df['avg_sentiment'], 
                             alpha=0.8, s=100, c='purple')
            axes[1, 1].set_xlabel('Average Confidence')
            axes[1, 1].set_ylabel('Average Sentiment')
            axes[1, 1].set_title('Confidence vs Sentiment')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def run_complete_pipeline(self, fetch_market_data_flag=True):
        # 1. Load and preprocess
        print("1. Loading and preprocessing data...")
        df = self.load_and_preprocess_data()
        original_df = df.copy()  # Keep for keyword analysis
        print(f"   Loaded {len(df)} headlines from {df['Date'].min()} to {df['Date'].max()}")
        
        # 2. Create daily aggregations
        print("2. Creating daily sentiment aggregations...")
        daily_df = self.daily_sentiment_aggregation(df)
        print(f"   Created daily data for {len(daily_df)} days")
        
        # 3. Add rolling features
        print("3. Adding rolling window features...")
        daily_df = self.create_rolling_features(daily_df)
        
        # 4. Add regime features
        print("4. Creating sentiment regime features...")
        daily_df = self.create_sentiment_features(daily_df)
        
        # 5. Add keyword features
        print("5. Adding keyword-based features...")
        daily_df = self.create_keyword_based_features(original_df, daily_df)
        
        # 6. Fetch market data and create targets
        print("6. Fetching market data and creating targets...")
        if fetch_market_data_flag:
            try:
                market_data = self.fetch_market_data(
                    daily_df['Date'].min(), 
                    daily_df['Date'].max()
                )
                daily_df = self.create_target_variables(daily_df, market_data)
                print("   Successfully merged with market data")
            except Exception as e:
                print(f"   Error fetching market data: {e}")
                print("   Using synthetic targets instead")
                daily_df = self.create_target_variables(daily_df, {})
        else:
            daily_df = self.create_target_variables(daily_df, {})
        
        # 7. Prepare features
        print("7. Preparing features for modeling...")
        X = self.prepare_features_for_modeling(daily_df)
        print(f"   Created feature matrix with {X.shape[1]} features")
        
        # 8. Train models for different targets
        results = {}
        targets = ['next_day_positive', 'next_3day_positive']
        min_samples_for_training = 5  # Reduced minimum for small datasets
        
        for target in targets:
            if target in daily_df.columns:
                print(f"\n8. Training models for {target}...")
                valid_idx = ~daily_df[target].isna()
                X_target = X[valid_idx]
                y_target = daily_df[target][valid_idx]
                
                if len(y_target) >= min_samples_for_training:
                    target_results = self.train_and_evaluate_models(X_target, y_target, 'classification')
                    results[target] = target_results
                    
                    # Print results
                    print(f"   Results for {target}:")
                    for name, result in target_results.items():
                        print(f"     {name}: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")
                else:
                    print(f"   Insufficient data for {target} ({len(y_target)} samples, need at least {min_samples_for_training})")
        
        # 9. Create dashboard
        print("\n9. Creating prediction dashboard...")
        try:
            if results:
                best_target = max(results.keys(), key=lambda k: max(results[k][model]['mean_score'] 
                                                                   for model in results[k].keys()))
                self.create_prediction_dashboard(daily_df, results[best_target], best_target)
            else:
                # Create simplified dashboard without model results
                self.create_simplified_dashboard(daily_df)
        except Exception as e:
            print(f"   Error creating dashboard: {e}")
            print("   Creating simplified visualization instead...")
            self.create_simplified_dashboard(daily_df)
        
        # 10. Summary and recommendations
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Data period: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
        print(f"Total days: {len(daily_df)}")
        print(f"Average headlines per day: {daily_df['headline_count'].mean():.1f}")
        print(f"Average daily sentiment: {daily_df['avg_sentiment'].mean():.3f}")
        
        # 10. Summary and recommendations
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Data period: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
        print(f"Total days: {len(daily_df)}")
        print(f"Average headlines per day: {daily_df['headline_count'].mean():.1f}")
        print(f"Average daily sentiment: {daily_df['avg_sentiment'].mean():.3f}")
        
        # Additional insights for small datasets
        print(f"\nSentiment Range: {daily_df['avg_sentiment'].min():.3f} to {daily_df['avg_sentiment'].max():.3f}")
        print(f"Most positive day: {daily_df.loc[daily_df['avg_sentiment'].idxmax(), 'Date'].strftime('%Y-%m-%d')} ({daily_df['avg_sentiment'].max():.3f})")
        print(f"Most negative day: {daily_df.loc[daily_df['avg_sentiment'].idxmin(), 'Date'].strftime('%Y-%m-%d')} ({daily_df['avg_sentiment'].min():.3f})")
        
        # Check if we have market data
        if 'daily_return' in daily_df.columns:
            valid_market_data = daily_df.dropna(subset=['daily_return'])
            if len(valid_market_data) > 0:
                correlation = daily_df['avg_sentiment'].corr(daily_df['daily_return'])
                print(f"\nSentiment-Market Correlation: {correlation:.3f}")
                print(f"Average market return: {daily_df['daily_return'].mean():.4f} ({daily_df['daily_return'].mean()*100:.2f}%)")
        
        if results:
            print(f"\nBest performing target: {best_target}")
            best_model_for_target = max(results[best_target].keys(), 
                                      key=lambda k: results[best_target][k]['mean_score'])
            best_score = results[best_target][best_model_for_target]['mean_score']
            print(f"Best model: {best_model_for_target} (Accuracy: {best_score:.4f})")
            
            # Feature importance
            best_model = results[best_target][best_model_for_target]['model']
            importance_df = self.get_feature_importance(best_model, self.feature_columns)
            if importance_df is not None:
                print(f"\nTop 5 most important features:")
                for i, row in importance_df.head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        else:
            print(f"\n⚠️  RECOMMENDATIONS FOR IMPROVING PREDICTIONS:")
            print(f"   • Collect more data (current: {len(daily_df)} days, recommended: 30+ days)")
            print(f"   • Current dataset is too small for reliable model training")
            print(f"   • Consider collecting headlines over a longer time period")
            print(f"   • Monitor sentiment patterns as shown in the dashboard")
        
        print(f"\n🔧 TECHNICAL NOTES:")
        print(f"   • Features created: {len(self.feature_columns)}")
        print(f"   • Data quality: {'Good' if daily_df['headline_count'].mean() > 10 else 'Low (few headlines per day)'}")
        print(f"   • Sentiment consistency: {'High' if daily_df['avg_sentiment'].std() < 0.2 else 'Variable'}")
        
        return {
            'daily_data': daily_df,
            'features': X,
            'results': results,
            'original_data': original_df,
            'summary_stats': {
                'total_days': len(daily_df),
                'avg_headlines_per_day': daily_df['headline_count'].mean(),
                'avg_sentiment': daily_df['avg_sentiment'].mean(),
                'sentiment_volatility': daily_df['avg_sentiment'].std(),
                'feature_count': len(self.feature_columns)
            }
        }