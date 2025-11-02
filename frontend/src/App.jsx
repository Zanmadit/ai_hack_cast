import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

function App() {
  const [graphs, setGraphs] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/graphs")
      .then(res => setGraphs(res.data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Forecast Dashboard</h1>
      {graphs.map((g, idx) => (
        <div key={idx} style={{ display: "flex", gap: "40px", marginBottom: 50 }}>
          <div style={{ flex: 1 }}>
            <h3>{g.file} — {g.metric} (History)</h3>
            <Plot
              data={JSON.parse(g.hist_graph).data}
              layout={JSON.parse(g.hist_graph).layout}
              config={{ responsive: true }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <h3>{g.file} — {g.metric} (Forecast {'>='} 2020)</h3>
            <Plot
              data={JSON.parse(g.forecast_graph).data}
              layout={JSON.parse(g.forecast_graph).layout}
              config={{ responsive: true }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
