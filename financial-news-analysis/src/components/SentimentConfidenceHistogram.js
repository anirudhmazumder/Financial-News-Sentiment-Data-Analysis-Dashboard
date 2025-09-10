import { ResponsiveContainer, XAxis, YAxis, Tooltip, BarChart, Bar } from "recharts";

const SentimentConfidenceHistogram = ({ data }) => {
  // Extract confidence values
  const values = data.map((item) => item.Confidence);

  // Define bin settings
  const numBins = 10;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binSize = (max - min) / numBins;

  // Initialize bins
  const bins = Array.from({ length: numBins }, (_, i) => ({
    bin: `${(min + i * binSize).toFixed(2)} - ${(min + (i + 1) * binSize).toFixed(2)}`,
    count: 0,
  }));

  // Fill bins
  values.forEach((v) => {
    let index = Math.floor((v - min) / binSize);
    if (index === numBins) index = numBins - 1; // Edge case for max value
    bins[index].count++;
  });

  return (
    <div className="p-6 max-w-2xl mx-auto bg-white shadow-lg rounded-xl">
      <h3 className="text-xl font-bold mb-4 text-center">Sentiment Confidence Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={bins}>
          <XAxis dataKey="bin" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SentimentConfidenceHistogram;
