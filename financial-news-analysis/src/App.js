import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { useState, useEffect, useCallback } from "react";
import Home from "./components/Home";
import NewsFeed from "./components/NewsFeed";
import DataAnalysis from "./components/DataAnalysis";

function App() {
  const [sentimentData, setSentimentData] = useState([]);
  const [dailySentimentIndexData, setDailySentimentIndexData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const API_BASE_URL = "http://127.0.0.1:8000";

  const fetchWithErrorHandling = async (url) => {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        if (response.status === 202) {
          // Analysis is running, return partial data
          setIsUpdating(true);
          throw new Error('Analysis is currently running. Showing cached data...');
        } else if (response.status === 404) {
          throw new Error('No data available. Analysis may not have completed yet.');
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }

      const result = await response.json();
      
      // Handle new response format with metadata
      if (result.data !== undefined) {
        setLastUpdated(result.last_updated);
        setIsUpdating(result.is_updating || false);
        return result.data;
      } else {
        // Fallback for old format
        return result;
      }
    } catch (err) {
      console.error(`Error fetching ${url}:`, err);
      
      // Don't throw error for analysis running - just show warning
      if (err.message.includes('Analysis is currently running')) {
        setError(err.message);
        return null; // Return null but don't stop the app
      }
      
      throw err;
    }
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [sentimentRes, indexRes] = await Promise.allSettled([
        fetchWithErrorHandling(`${API_BASE_URL}/sentiment`),
        fetchWithErrorHandling(`${API_BASE_URL}/daily-sentiment-index`)
      ]);

      // Handle sentiment data
      if (sentimentRes.status === 'fulfilled' && sentimentRes.value) {
        setSentimentData(sentimentRes.value);
      } else if (sentimentRes.status === 'rejected') {
        console.error('Sentiment data fetch failed:', sentimentRes.reason.message);
      }

      // Handle daily index data
      if (indexRes.status === 'fulfilled' && indexRes.value) {
        setDailySentimentIndexData(indexRes.value);
      } else if (indexRes.status === 'rejected') {
        console.error('Daily index data fetch failed:', indexRes.reason.message);
      }

      // If both failed, show error
      if (sentimentRes.status === 'rejected' && indexRes.status === 'rejected') {
        setError('Failed to load data. Please try refreshing the page.');
      }

    } catch (err) {
      console.error('Unexpected error:', err);
      setError('An unexpected error occurred. Please try refreshing the page.');
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  const refreshData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        setIsUpdating(true);
        setError('Analysis refresh started. Data will update automatically...');
        // Wait a bit then fetch new data
        setTimeout(() => {
          fetchData();
        }, 5000);
      } else {
        throw new Error(`Refresh failed: ${response.status}`);
      }
    } catch (err) {
      setError(`Failed to refresh: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL, fetchData]);

  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      if (response.ok) {
        const status = await response.json();
        setIsUpdating(status.is_running);
        if (status.last_updated) {
          setLastUpdated(status.last_updated);
        }
        
        // If analysis just finished, refetch data
        if (!status.is_running && isUpdating) {
          setTimeout(() => fetchData(), 2000);
        }
      }
    } catch (err) {
      console.error('Error checking status:', err);
    }
  }, [API_BASE_URL, fetchData, isUpdating]);

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Poll for status updates when analysis is running
  useEffect(() => {
    let pollInterval;
    
    if (isUpdating) {
      pollInterval = setInterval(() => {
        checkStatus();
      }, 10000); // Check every 10 seconds
    }

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [isUpdating, checkStatus]);

  return (
    <Router>
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <nav className="bg-gray-800 text-white p-6 flex flex-col gap-6 min-w-[220px]">
          <h2 className="text-2xl font-semibold mb-4 tracking-wide">Dashboard</h2>
          
          {/* Status indicator */}
          <div className="mb-4 text-sm">
            {isUpdating && (
              <div className="flex items-center gap-2 text-yellow-400">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-yellow-400"></div>
                Analysis running...
              </div>
            )}
            {lastUpdated && !isUpdating && (
              <div className="text-gray-400">
                Updated: {new Date(lastUpdated).toLocaleTimeString()}
              </div>
            )}
          </div>

          <Link className="text-lg hover:text-yellow-400 transition-colors" to="/">Home</Link>
          <Link className="text-lg hover:text-yellow-400 transition-colors" to="/headlines">Headlines</Link>
          <Link className="text-lg hover:text-yellow-400 transition-colors" to="/data-analysis">Data Analysis</Link>
          
          {/* Control buttons */}
          <div className="mt-8 space-y-2">
            <button
              onClick={fetchData}
              disabled={loading}
              className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-sm transition-colors"
            >
              {loading ? 'Loading...' : 'Refresh Data'}
            </button>
            
            <button
              onClick={refreshData}
              disabled={isUpdating || loading}
              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-sm transition-colors"
            >
              {isUpdating ? 'Running...' : 'Force Update'}
            </button>
          </div>
        </nav>

        {/* Main Content */}
        <div className="p-4 flex-1">
          {/* Error/Status Banner */}
          {error && (
            <div className={`mb-4 p-4 rounded-lg ${
              error.includes('running') || error.includes('refresh started') 
                ? 'bg-yellow-100 border-yellow-400 text-yellow-800' 
                : 'bg-red-100 border-red-400 text-red-800'
            } border`}>
              <div className="flex items-center justify-between">
                <span>{error}</span>
                <button
                  onClick={() => setError(null)}
                  className="text-lg font-bold hover:opacity-70"
                >
                  ×
                </button>
              </div>
            </div>
          )}

          {/* Loading indicator for initial load */}
          {loading && sentimentData.length === 0 && dailySentimentIndexData.length === 0 && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading financial data...</p>
                {isUpdating && <p className="text-sm text-yellow-600 mt-2">Running analysis...</p>}
              </div>
            </div>
          )}

          {/* Routes */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route 
              path="/headlines" 
              element={
                <NewsFeed 
                  sentimentData={sentimentData}
                  loading={loading && sentimentData.length === 0}
                />
              } 
            />
            <Route 
              path="/data-analysis" 
              element={
                <DataAnalysis 
                  headlineSentimentData={sentimentData}
                  dailySentimentIndexData={dailySentimentIndexData}
                  loading={loading && sentimentData.length === 0 && dailySentimentIndexData.length === 0}
                  isUpdating={isUpdating}
                  lastUpdated={lastUpdated}
                />
              } 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;