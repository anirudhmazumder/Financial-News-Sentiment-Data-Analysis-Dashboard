import React, { useEffect, useState } from "react";

const DataLoader = ({ children }) => {
    const [sentimentData, setSentimentData] = useState([]);
    const [sentimentIndex, setSentimentIndex] = useState([]);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
    const fetchData = async () => {
      try {
        const [sentimentRes, indexRes] = await Promise.all([
          fetch("http://127.0.0.1:8000/sentiment").then((res) => res.json()),
          fetch("http://127.0.0.1:8000/daily-sentiment-index").then((res) =>
            res.json()
          ),
        ]);

        setSentimentData(sentimentRes);
        setSentimentIndex(indexRes);
      } catch (err) {
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <p className="text-center">Loading data...</p>;

  return React.cloneElement(children, { sentimentData, sentimentIndex });
};

export default DataLoader;
