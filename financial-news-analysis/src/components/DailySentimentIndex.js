import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DailySentimentIndex = ({ data }) => {
    return (
        <div className="p-6 max-w-4xl mx-auto bg-white shadow-lg rounded-xl">
            <h3 className="text-xl font-bold mb-4 text-center">Daily Sentiment Index</h3>
            {data.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="Time" />
                        <YAxis dataKey="Daily_Sentiment_Index" />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="Daily_Sentiment_Index" fill="#8884d8" name="Sentiment Index" />
                    </BarChart>
                </ResponsiveContainer>
            ) : (
                <p className="text-center text-gray-500">No data available</p>
            )}
        </div>
    );
};

export default DailySentimentIndex;