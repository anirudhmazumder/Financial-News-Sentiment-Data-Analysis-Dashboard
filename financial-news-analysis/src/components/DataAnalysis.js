import SentimentChart from './SentimentChart';
import DailySentimentIndex from './DailySentimentIndex';
import SentimentConfidenceHistogram from './SentimentConfidenceHistogram';

const DataAnalysis = ({ headlineSentimentData, dailySentimentIndexData }) => {
  return (
    <div className="p-6">
      <h1 className="text-4xl font-bold mb-8 text-center">Data Analysis</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Sentiment chart (left on md+) */}
        <div className="col-span-1">
          <SentimentChart data={headlineSentimentData} />
        </div>

        {/* Histogram (right on md+) */}
        <div className="col-span-1">
          <SentimentConfidenceHistogram data={headlineSentimentData} />
        </div>

        {/* Daily Sentiment Index — full width row */}
        <div className="col-span-1 md:col-span-2">
          <DailySentimentIndex data={dailySentimentIndexData} />
        </div>
      </div>
    </div>
  );
};

export default DataAnalysis;
