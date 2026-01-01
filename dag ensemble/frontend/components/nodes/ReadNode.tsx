import React, { memo, useState, useEffect, useRef } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

const ReadNode = ({ data, id }: NodeProps) => {
    const [showExpanded, setShowExpanded] = useState(false);
    const [fullData, setFullData] = useState<any[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [availableFiles, setAvailableFiles] = useState<string[]>([]);
    const [selectedFile, setSelectedFile] = useState<string>(data.filename || 'data_storage.parquet');
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<number>(Date.now());

    // Sync selected file with node data (internal local state mainly, as we don't have updateNodeData readily available in this context without custom hooks, 
    // but for polling it works with local state)

    // 1. Fetch Available Files
    useEffect(() => {
        fetch('http://localhost:8001/data-storage-files')
            .then(res => res.json())
            .then(data => {
                if (data.files) setAvailableFiles(data.files);
            })
            .catch(err => console.error("ReadNode: Failed to fetch files", err));
    }, []);

    // 2. Poll Data
    useEffect(() => {
        if (!autoRefresh) return;

        const fetchData = () => {
            // Only fetch if we have a file
            if (!selectedFile) return;

            setLoading(true);
            fetch(`http://localhost:8001/storage/content?filename=${selectedFile}&rows=50`)
                .then(res => res.json())
                .then(pkg => {
                    if (pkg.data && Array.isArray(pkg.data)) {
                        setFullData(pkg.data);
                        setLastUpdated(Date.now());
                    }
                })
                .catch(err => console.error("ReadNode: Failed to fetch content", err))
                .finally(() => setLoading(false));
        };

        fetchData(); // Initial fetch
        const interval = setInterval(fetchData, 2000); // Poll every 2s

        return () => clearInterval(interval);
    }, [selectedFile, autoRefresh]);

    const formatTime = (ts: any) => {
        if (!ts) return '';
        try {
            // Check if unix timestamp (seconds or ms)
            if (typeof ts === 'number') {
                // heuristic: if small, seconds. if huge, ms.
                if (ts < 30000000000) return new Date(ts * 1000).toLocaleTimeString();
                return new Date(ts).toLocaleTimeString();
            }
            return new Date(ts).toLocaleTimeString();
        } catch { return ts; }
    }

    return (
        <div className="p-4 border border-fuchsia-200 rounded-lg bg-white shadow-md w-80 ring-1 ring-fuchsia-50">
            {/* Header */}
            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-fuchsia-100 flex items-center justify-center text-fuchsia-600 font-bold text-xs">R</div>
                    <div className="font-bold text-sm text-zinc-900">Read Storage</div>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${autoRefresh ? 'bg-green-50 text-green-600 border-green-200' : 'bg-gray-50 text-gray-400 border-gray-200'}`}
                    >
                        {autoRefresh ? 'LIVE' : 'PAUSED'}
                    </button>
                </div>
            </div>

            {/* File Selection */}
            <div className="mb-3">
                <label className="text-[10px] uppercase text-zinc-400 font-semibold block mb-1">Target File</label>
                <select
                    className="w-full text-xs border border-zinc-300 rounded px-2 py-1 focus:outline-none focus:border-fuchsia-500 bg-white"
                    value={selectedFile}
                    onChange={(e) => {
                        setSelectedFile(e.target.value);
                        data.filename = e.target.value; // Try to persist locally
                    }}
                >
                    {availableFiles.map(f => (
                        <option key={f} value={f}>{f}</option>
                    ))}
                </select>
            </div>

            {/* Data Preview */}
            <div className="space-y-2">
                <div className="flex justify-between items-center text-[10px] text-zinc-500 uppercase font-semibold">
                    <span>Preview</span>
                    <button
                        onClick={() => setShowExpanded(true)}
                        className="text-xs bg-fuchsia-50 text-fuchsia-600 px-2 py-0.5 rounded hover:bg-fuchsia-100 transition-colors"
                    >
                        Expand View
                    </button>
                </div>

                <div className="bg-zinc-50 border border-zinc-200 rounded p-2 text-xs font-mono h-48 overflow-y-auto overflow-x-auto relative">
                    {loading && <div className="absolute top-1 right-1 w-2 h-2 bg-fuchsia-400 rounded-full animate-ping"></div>}

                    {fullData && fullData.length > 0 ? (
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-zinc-200 text-zinc-500">
                                    <th className="py-1 px-1 font-normal">Time</th>
                                    {Object.keys(fullData[0])
                                        .filter(k => k !== 'timestamp')
                                        .slice(0, 2) // Show first 2 data cols
                                        .map(k => (
                                            <th key={k} className="py-1 px-1 font-normal">{k.split('_').slice(-1)[0]}</th>
                                        ))}
                                </tr>
                            </thead>
                            <tbody>
                                {[...fullData].reverse().slice(0, 20).map((row: any, idx: number) => (
                                    <tr key={idx} className="border-b border-zinc-100 last:border-0 hover:bg-zinc-100">
                                        <td className="py-1 px-1 text-zinc-400 whitespace-nowrap text-[10px]">
                                            {formatTime(row.timestamp)}
                                        </td>
                                        {Object.keys(row)
                                            .filter(k => k !== 'timestamp')
                                            .slice(0, 2)
                                            .map(k => (
                                                <td key={k} className="py-1 px-1 font-medium text-zinc-700">
                                                    {typeof row[k] === 'number' ? row[k].toFixed(4) : String(row[k])}
                                                </td>
                                            ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="text-zinc-400 italic flex items-center justify-center h-full">
                            {loading ? "Loading..." : "No Data / Empty File"}
                        </div>
                    )}
                </div>
                <div className="text-[10px] text-right text-zinc-300">
                    Updated: {new Date(lastUpdated).toLocaleTimeString()}
                </div>
            </div>

            {/* Expanded Modal */}
            {showExpanded && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setShowExpanded(false)}>
                    <div className="bg-white rounded-xl shadow-2xl w-[90vw] h-[80vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-zinc-100 flex justify-between items-center bg-zinc-50">
                            <h3 className="font-bold text-lg text-zinc-800">
                                {selectedFile} <span className="font-normal text-sm text-zinc-500 ml-2">({fullData?.length || 0} rows)</span>
                            </h3>
                            <button onClick={() => setShowExpanded(false)} className="text-zinc-400 hover:text-zinc-600 text-xl">✕</button>
                        </div>
                        <div className="flex-grow overflow-auto p-4 bg-white">
                            <table className="w-full text-xs font-mono text-left border-collapse">
                                <thead className="sticky top-0 bg-zinc-100 shadow-sm z-10">
                                    <tr>
                                        {fullData && fullData.length > 0 && Object.keys(fullData[0]).map(k => (
                                            <th key={k} className="p-2 border-b border-zinc-300 font-bold text-zinc-600">{k}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {(fullData || []).slice().reverse().map((row: any, idx: number) => (
                                        <tr key={idx} className="hover:bg-fuchsia-50 border-b border-zinc-100 transition-colors">
                                            {Object.keys(row).map(k => (
                                                <td key={k} className="p-2 text-zinc-800 border-r border-zinc-50 last:border-0">
                                                    {k === 'timestamp'
                                                        ? formatTime(row[k])
                                                        : (typeof row[k] === 'number' ? row[k].toFixed(5) : String(row[k]))}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default memo(ReadNode);
