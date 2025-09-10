import React from 'react';
import { PieChart, Pie, Cell, Legend, Tooltip } from 'recharts';

const SentimentChart = ({ data }) => {
    const sentimentDistributionCount = data.reduce((acc, item) => {
        acc[item.Sentiment] = (acc[item.Sentiment] || 0) + 1;
        return acc;
    }, {Positive: 0, Negative: 0, Neutral: 0});

    const chartData = [
        { name: 'Positive', value: sentimentDistributionCount.Positive },
        { name: 'Negative', value: sentimentDistributionCount.Negative },
        { name: 'Neutral', value: sentimentDistributionCount.Neutral }
    ];

    const COLORS = ["#00C49F", "#8884d8", "#FF4C4C"];

    return (
        <div className="p-6 max-w-md mx-auto bg-white shadow-lg rounded-xl">
            <h3 className="text-xl font-bold mb-4 text-center">Sentiment Distributions</h3>
            <PieChart width={400} height={300}>
                <Pie
                    data={chartData}
                    cx={"50%"}
                    cy={"50%"}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label
                >
                    {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index]} />
                    ))}
                </Pie>
                <Tooltip />
                <Legend />
            </PieChart>
        </div>
    );
}

export default SentimentChart;