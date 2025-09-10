import { useState } from 'react';

const NewsFeed = ( {sentimentData} ) => {
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 5;

    const totalPages = Math.ceil(sentimentData.length / itemsPerPage);
    const currentData = sentimentData.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    return (
        <div>
            <h1 className="text-4xl font-bold mb-4 text-center">Financial News Headlines</h1>
            <ul>
                {currentData.map((item, index) => {
                    const globalIndex = (currentPage - 1) * itemsPerPage + index + 1;
                    return (
                        <li key={index} className="p-4 mb-4 border rounded-lg shadow hover:shadow-lg transition-shadow">
                            <strong className="text-lg">
                                {globalIndex}. {item.Headline}
                            </strong>
                            <div className="flex flex-col mt-1 text-sm gap-1">
                                <span>
                                    Sentiment:{" "}
                                    <span
                                        className={`font-bold ${
                                            item.Sentiment === "Positive"
                                                ? "text-green-500"
                                                : item.Sentiment === "Negative"
                                                ? "text-red-500"
                                                : "text-gray-500"
                                        }`}
                                    >
                                        {item.Sentiment}
                                    </span>
                                </span>
                                <span>Date: {item.Time}</span>
                                <span>Confidence: {item.Confidence}</span>
                            </div>
                        </li>
                    );
                })}
            </ul>
            <div className="flex justify-center mt-4">
                <button
                    className="px-4 py-2 bg-gray-300 rounded mr-2"
                    onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                >
                    Previous
                </button>
                <button
                    className="px-4 py-2 bg-gray-300 rounded"
                    onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                >
                    Next
                </button>
            </div>
        </div>
    );
};

export default NewsFeed;