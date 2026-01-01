import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const PAIRS = ['XBT/USD', 'ETH/USD', 'SOL/USD', 'USDT/USD', 'XRP/USD', 'ADA/USD', 'DOT/USD', 'LTC/USD'];
const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

const LiveDataFeedNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [showExpanded, setShowExpanded] = React.useState(false);

    // Default configuration if not present
    const pair = data.pair || 'XBT/USD';
    const timeframe = data.timeframe || '1m';

    const updateData = useCallback((key: string, value: string) => {
        setNodes((nodes) =>
            nodes.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, [key]: value } };
                }
                return node;
            })
        );
    }, [id, setNodes]);

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-72 transition-all duration-200 ${data.isExecuting ? 'border-red-500 ring-2 ring-red-200 shadow-red-100' : 'border-red-200'
            }`}>
            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-red-100 flex items-center justify-center text-red-600 font-bold text-xs animate-pulse">LIVE</div>
                    <div className="font-bold text-sm text-zinc-900">Kraken Feed</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-red-100 text-red-700 border-red-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}

            </div>

            {/* Manual Configuration */}
            <div className="space-y-3 mb-4">
                <div className="text-[10px] text-zinc-500 font-semibold uppercase">Configuration</div>
                <div className="bg-zinc-50 border border-zinc-100 rounded p-2 space-y-2">
                    {/* Pair Selection */}
                    <div className="flex flex-col gap-1">
                        <label className="text-xs text-zinc-500 font-medium">Pair</label>
                        <select
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-red-400"
                            value={pair}
                            onChange={(e) => updateData('pair', e.target.value)}
                        >
                            {PAIRS.map(p => (
                                <option key={p} value={p}>{p}</option>
                            ))}
                        </select>
                    </div>

                    {/* Timeframe Selection */}
                    <div className="flex flex-col gap-1">
                        <label className="text-xs text-zinc-500 font-medium">Timeframe</label>
                        <select
                            className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-700 outline-none focus:border-red-400"
                            value={timeframe}
                            onChange={(e) => updateData('timeframe', e.target.value)}
                        >
                            {TIMEFRAMES.map(t => (
                                <option key={t} value={t}>{t}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            <div className="space-y-2">
                <div className="text-[10px] text-zinc-500 uppercase font-semibold flex justify-between items-center">
                    <span>Dataset Snapshot</span>
                    <button
                        onClick={() => setShowExpanded(true)}
                        className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded hover:bg-blue-100 transition-colors"
                    >
                        Expand
                    </button>
                </div>

                <div className="bg-zinc-50 border border-zinc-200 rounded p-2 text-xs font-mono h-48 overflow-y-auto overflow-x-auto">
                    {data.feedSnapshot && data.feedSnapshot.length > 0 ? (
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-zinc-200 text-zinc-500">
                                    <th className="py-1 px-1 font-normal">Time</th>
                                    {Object.keys(data.feedSnapshot[0])
                                        .filter(k => k !== 'timestamp')
                                        .slice(0, 3) // Show first 3 cols only to save space
                                        .map(k => (
                                            <th key={k} className="py-1 px-1 font-normal">{k.split('_').slice(-1)[0]}</th> // Show last part of name
                                        ))}
                                </tr>
                            </thead>
                            <tbody>
                                {[...data.feedSnapshot].reverse().map((row: any, idx: number) => (
                                    <tr key={idx} className="border-b border-zinc-100 last:border-0 hover:bg-zinc-100">
                                        <td className="py-1 px-1 text-zinc-400 whitespace-nowrap">
                                            {formatTime(row.timestamp)}
                                        </td>
                                        {Object.keys(row)
                                            .filter(k => k !== 'timestamp')
                                            .slice(0, 3)
                                            .map(k => (
                                                <td key={k} className="py-1 px-1 font-medium text-zinc-700">
                                                    {typeof row[k] === 'object' && row[k] !== null
                                                        ? JSON.stringify(row[k])
                                                        : (typeof row[k] === 'number' ? row[k].toFixed(2) : row[k])}
                                                </td>
                                            ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="text-zinc-400 italic flex items-center justify-center h-full">
                            Waiting for data...
                        </div>
                    )}
                </div>
            </div>

            <Handle type="source" position={Position.Right} className="!bg-red-500 !w-3 !h-3" />

            {/* Expanded Modal */}
            {showExpanded && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setShowExpanded(false)}>
                    <div className="bg-white rounded-xl shadow-2xl w-[90vw] h-[80vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-zinc-100 flex justify-between items-center bg-zinc-50">
                            <h3 className="font-bold text-lg text-zinc-800">Full Dataset View ({pair} - {timeframe})</h3>
                            <button onClick={() => setShowExpanded(false)} className="text-zinc-400 hover:text-zinc-600">✕</button>
                        </div>
                        <div className="flex-grow overflow-auto p-4 bg-white">
                            <FullDatasetViewer pair={pair} timeframe={timeframe} />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Subcomponent for fetching and displaying full dataset
const FullDatasetViewer = ({ pair, timeframe }: { pair: string, timeframe: string }) => {
    const [dataset, setDataset] = React.useState<any[]>([]);
    const [loading, setLoading] = React.useState(true);

    React.useEffect(() => {
        setLoading(true);
        // Add timestamp to prevent caching
        fetch(`http://localhost:8001/dataset?pair=${pair}&timeframe=${timeframe}&t=${Date.now()}`)
            .then(res => res.json())
            .then(data => {
                setDataset(data.data || []);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to fetch dataset", err);
                setLoading(false);
            });
    }, [pair, timeframe]);

    if (loading) return <div className="flex justify-center items-center h-full text-zinc-400">Loading full history...</div>;
    if (dataset.length === 0) return <div className="flex justify-center items-center h-full text-zinc-400">No data available.</div>;

    const keys = Object.keys(dataset[0]).filter(k => k !== 'timestamp');

    return (
        <table className="w-full text-xs font-mono text-left border-collapse">
            <thead className="sticky top-0 bg-zinc-100 shadow-sm z-10">
                <tr>
                    <th className="p-2 border-b border-zinc-300 font-bold text-zinc-600">Timestamp</th>
                    {keys.map(k => (
                        <th key={k} className="p-2 border-b border-zinc-300 font-bold text-zinc-600">{k}</th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {[...dataset].reverse().map((row, idx) => (
                    <tr key={idx} className="hover:bg-blue-50 border-b border-zinc-100">
                        <td className="p-2 text-zinc-500 whitespace-nowrap">{new Date(isNaN(Number(row.timestamp)) ? row.timestamp : Number(row.timestamp) * 1000).toLocaleString()}</td>
                        {keys.map(k => (
                            <td key={k} className="p-2 text-zinc-800">
                                {typeof row[k] === 'object' && row[k] !== null
                                    ? JSON.stringify(row[k])
                                    : (typeof row[k] === 'number' ? row[k].toFixed(2) : row[k])}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

const formatTime = (ts: any) => {
    if (!ts) return '';
    try {
        // If it's a large int/float string, assume seconds
        if (!isNaN(ts)) {
            return new Date(Number(ts) * 1000).toLocaleTimeString();
        }
        return new Date(ts).toLocaleTimeString();
    } catch { return ts; }
}

export default memo(LiveDataFeedNode);
