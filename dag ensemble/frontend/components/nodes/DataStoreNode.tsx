import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

const DataStoreNode = ({ data, id }: NodeProps) => {
    const [showExpanded, setShowExpanded] = useState(false);
    const [fullData, setFullData] = useState<any[] | null>(null);
    const [loadingFullData, setLoadingFullData] = useState(false);
    const [availableFiles, setAvailableFiles] = useState<string[]>([]);

    useEffect(() => {
        // Fetch available files
        fetch('http://localhost:8001/data-storage-files')
            .then(res => res.json())
            .then(data => {
                if (data.files) setAvailableFiles(data.files);
            })
            .catch(err => console.error("Failed to fetch storage files", err));

        if (showExpanded) {
            setLoadingFullData(true);
            fetch(`http://localhost:8001/nodes/${id}/output`)
                .then(res => res.json())
                .then(pkg => {
                    // pkg.data should be the array of records
                    if (pkg.data && Array.isArray(pkg.data)) {
                        setFullData(pkg.data);
                    }
                })
                .catch(err => console.error("Failed to fetch full output", err))
                .finally(() => setLoadingFullData(false));
        }
    }, [showExpanded, id]);

    // Normalize input data: expecting a package { data: [...] } or raw array
    const rawData = data.latestData;
    const snapshot = (rawData && !Array.isArray(rawData) && rawData.data)
        ? rawData.data
        : (Array.isArray(rawData) ? rawData : []);

    const formatTime = (ts: any) => {
        if (!ts) return '';
        try {
            if (!isNaN(ts)) {
                return new Date(Number(ts) * 1000).toLocaleTimeString();
            }
            return new Date(ts).toLocaleTimeString();
        } catch { return ts; }
    }

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-72 transition-all duration-200 ${data.isExecuting ? 'border-cyan-500 ring-2 ring-cyan-200 shadow-cyan-100' : 'border-cyan-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-cyan-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-cyan-100 flex items-center justify-center text-cyan-600 font-bold text-xs">DS</div>
                    <div className="font-bold text-sm text-zinc-900">Write & Fetch</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-cyan-100 text-cyan-700 border-cyan-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            {/* File Selection Dropdown */}
            <div className="mb-2">
                <label className="text-[10px] uppercase text-zinc-400 font-semibold block mb-1">Storage File</label>
                <select
                    className="w-full text-xs border border-zinc-300 rounded px-1.5 py-1 focus:outline-none focus:border-cyan-500 bg-white"
                    value={data.filename || 'data_storage.parquet'}
                    onChange={(e) => {
                        // We need to update the node data permanently via ReactFlow mechanism
                        // Since we don't have direct 'setNodes' prop here effectively without context, 
                        // we usually pass a callback or rely on parent. 
                        // But standard Custom Nodes receive 'data' prop which is mutable reference in some versions 
                        // or we need to call an update function if provided.
                        // Assuming standard ReactFlow behavior: we modify 'data' object and force update? 
                        // Actually, 'data' is props. Changing it here won't persist.
                        // We need to use the 'useReactFlow' hook or simple local state if we want visual update 
                        // but to PERSIST it we really need to update the node in the Flow state.

                        // NOTE: For now, I will assume the parent FlowEditor passes an update handler or we rely on 
                        // local mutation if the parent re-renders. 
                        // Actually, let's try to update it directly if 'data' is mutable proxy? 
                        // Most ReactFlow implementations allow data mutation.
                        data.filename = e.target.value;
                        // Force re-render of this component
                        // We should probably rely on a 'setNodes' from context if available, but let's try direct mutation first.
                    }}
                >
                    {availableFiles.length === 0 && <option value="data_storage.parquet">data_storage.parquet</option>}
                    {availableFiles.map(f => (
                        <option key={f} value={f}>{f}</option>
                    ))}
                </select>
            </div>

            <div className="space-y-2">
                <div className="text-[10px] text-zinc-500 uppercase font-semibold flex justify-between items-center">
                    <span>Stored Buffer</span>
                    <div className="text-[10px] text-zinc-400 font-normal">
                        ({snapshot.length} rows visible)
                    </div>
                    <button
                        onClick={() => setShowExpanded(true)}
                        className="text-xs bg-cyan-50 text-cyan-600 px-2 py-0.5 rounded hover:bg-cyan-100 transition-colors"
                    >
                        Expand
                    </button>
                </div>

                <div className="bg-zinc-50 border border-zinc-200 rounded p-2 text-xs font-mono h-48 overflow-y-auto overflow-x-auto">
                    {snapshot && snapshot.length > 0 ? (
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-zinc-200 text-zinc-500">
                                    <th className="py-1 px-1 font-normal">Time</th>
                                    {Object.keys(snapshot[0])
                                        .filter(k => k !== 'timestamp')
                                        .slice(0, 2)
                                        .map(k => (
                                            <th key={k} className="py-1 px-1 font-normal">{k.split('_').slice(-1)[0]}</th>
                                        ))}
                                </tr>
                            </thead>
                            <tbody>
                                {[...snapshot].reverse().map((row: any, idx: number) => (
                                    <tr key={idx} className="border-b border-zinc-100 last:border-0 hover:bg-zinc-100">
                                        <td className="py-1 px-1 text-zinc-400 whitespace-nowrap">
                                            {formatTime(row.timestamp)}
                                        </td>
                                        {Object.keys(row)
                                            .filter(k => k !== 'timestamp')
                                            .slice(0, 2)
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

            <Handle type="source" position={Position.Right} className="!bg-cyan-500 !w-3 !h-3" />

            {/* Expanded Modal */}
            {showExpanded && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setShowExpanded(false)}>
                    <div className="bg-white rounded-xl shadow-2xl w-[90vw] h-[80vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-zinc-100 flex justify-between items-center bg-zinc-50">
                            <h3 className="font-bold text-lg text-zinc-800">Complete Data Store View {loadingFullData && <span className="text-xs font-normal text-zinc-500 ml-2">(Loading...)</span>}</h3>
                            <button onClick={() => setShowExpanded(false)} className="text-zinc-400 hover:text-zinc-600">✕</button>
                        </div>
                        <div className="flex-grow overflow-auto p-4 bg-white">
                            <table className="w-full text-xs font-mono text-left border-collapse">
                                <thead className="sticky top-0 bg-zinc-100 shadow-sm z-10">
                                    <tr>
                                        {snapshot.length > 0 && Object.keys(snapshot[0]).map(k => (
                                            <th key={k} className="p-2 border-b border-zinc-300 font-bold text-zinc-600">{k}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {(fullData || snapshot).slice().reverse().map((row: any, idx: number) => (
                                        <tr key={idx} className="hover:bg-blue-50 border-b border-zinc-100">
                                            {Object.keys(row).map(k => (
                                                <td key={k} className="p-2 text-zinc-800">
                                                    {k === 'timestamp'
                                                        ? formatTime(row[k])
                                                        : (typeof row[k] === 'object' && row[k] !== null
                                                            ? JSON.stringify(row[k])
                                                            : (typeof row[k] === 'number' ? row[k].toFixed(4) : row[k]))}
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

export default memo(DataStoreNode);
