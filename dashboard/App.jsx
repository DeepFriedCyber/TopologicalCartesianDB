
import { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, Button, Input, Select, MenuItem, FormControl, InputLabel, Alert } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

function parseFile(file, setData, setColumns) {
  const reader = new FileReader();
  reader.onload = (e) => {
    let result;
    if (file.name.endsWith('.json')) {
      result = JSON.parse(e.target.result);
      if (Array.isArray(result)) {
        setData(result);
        setColumns(Object.keys(result[0] || {}));
      } else if (result.data && Array.isArray(result.data)) {
        setData(result.data);
        setColumns(result.columns || Object.keys(result.data[0] || {}));
      }
    } else if (file.name.endsWith('.csv')) {
      // Simple CSV parser (for demo)
      const [header, ...rows] = e.target.result.split('\n').filter(Boolean);
      const cols = header.split(',');
      const data = rows.map(row => Object.fromEntries(row.split(',').map((v, i) => [cols[i], v])));
      setData(data);
      setColumns(cols);
    }
  };
  reader.readAsText(file);
}

export default function App() {
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [selectedDb, setSelectedDb] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('throughput');
  const [filename, setFilename] = useState('');
  const [error, setError] = useState('');

  // Auto-load latest results from backend
  useEffect(() => {
    fetch('/api/latest-results')
      .then(res => {
        if (!res.ok) throw new Error('No results found');
        return res.json();
      })
      .then(json => {
        setData(json.data);
        setColumns(json.columns);
        setFilename(json.filename);
        setError('');
      })
      .catch(() => setError('No results found in backend. Please upload a file.'));
  }, []);

  // Extract unique DBs/datasets/dimensions for filtering
  const dbs = Array.from(new Set(data.map(row => row.database || row.db || 'Unknown')));
  const datasets = Array.from(new Set(data.map(row => row.dataset || 'Unknown')));
  const dimensions = Array.from(new Set(data.map(row => row.dimension || row.dim || 'Unknown')));

  // Filtered data
  const filtered = data.filter(row =>
    (!selectedDb || (row.database || row.db) === selectedDb)
  );

  // Chart data
  const chartData = filtered.map(row => ({
    name: row.dataset || row.db || row.database || 'Unknown',
    throughput: Number(row.index_throughput || row.throughput || 0),
    qps: Number(row.qps || row.search_qps || 0),
    latency: Number(row.latency || row.search_latency || 0)
  }));

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>Vector DB Benchmark Dashboard</Typography>
      {filename && <Alert severity="info" sx={{ mb: 2 }}>Loaded: {filename}</Alert>}
      {error && <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert>}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Input type="file" inputProps={{ accept: '.json,.csv' }} onChange={e => {
          if (e.target.files[0]) parseFile(e.target.files[0], setData, setColumns);
        }} />
        <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Database</InputLabel>
            <Select value={selectedDb} label="Database" onChange={e => setSelectedDb(e.target.value)}>
              <MenuItem value="">All</MenuItem>
              {dbs.map(db => <MenuItem key={db} value={db}>{db}</MenuItem>)}
            </Select>
          </FormControl>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Metric</InputLabel>
            <Select value={selectedMetric} label="Metric" onChange={e => setSelectedMetric(e.target.value)}>
              <MenuItem value="throughput">Throughput</MenuItem>
              <MenuItem value="qps">QPS</MenuItem>
              <MenuItem value="latency">Latency</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6">Interactive Chart</Typography>
        <ResponsiveContainer width="100%" height={350}>
          {selectedMetric === 'latency' ? (
            <LineChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="latency" stroke="#8884d8" />
            </LineChart>
          ) : (
            <BarChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={selectedMetric} fill="#1976d2" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </Paper>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6">Raw Results Table</Typography>
        <Box sx={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {columns.map(col => <th key={col} style={{ border: '1px solid #ccc', padding: 4 }}>{col}</th>)}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => (
                <tr key={i}>
                  {columns.map(col => <td key={col} style={{ border: '1px solid #eee', padding: 4 }}>{row[col]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </Box>
      </Paper>
    </Container>
  );
}
