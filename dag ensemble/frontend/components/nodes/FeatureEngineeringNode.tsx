
import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const FeatureEngineeringNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [scripts, setScripts] = useState<string[]>([]);
    const [selectedScript, setSelectedScript] = useState(data.scriptName || '');
    const [showExpanded, setShowExpanded] = useState(false);
    const [fullData, setFullData] = useState<any[] | null>(null);
    const [loadingFullData, setLoadingFullData] = useState(false);

    // Fetch full data when expanded
    useEffect(() => {
        if (showExpanded) {
            setLoadingFullData(true);
            fetch(`http://localhost:8001/nodes/${id}/output`)
                .then(res => res.json())
                .then(pkg => {
                    if (pkg.data && Array.isArray(pkg.data)) {
                        setFullData(pkg.data);
                    }
                })
                .catch(err => console.error("Failed to fetch full output", err))
                .finally(() => setLoadingFullData(false));
        }
    }, [showExpanded, id]);

    // Fetch available scripts on mount
    useEffect(() => {
        fetch('http://localhost:8001/feature-scripts')
            .then(res => res.json())
            .then(data => {
                setScripts(data.scripts || []);
                // If current selection is invalid, reset? Or keep it maybe it's custom.
            })
            .catch(err => console.error("Failed to fetch scripts", err));
    }, []);

    const handleScriptChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newVal = e.target.value;
        setSelectedScript(newVal);

        // Update global node state so backend knows
        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, scriptName: newVal } };
                }
                return node;
            })
        );
    };

    // [FIX] data.latestData might be a raw array OR a package dict { data: [...], ... }
    const rawData = data.latestData;
    const snapshot = (rawData && !Array.isArray(rawData) && rawData.data)
        ? rawData.data
        : (Array.isArray(rawData) ? rawData : []);

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-72 transition-all duration-200 ${data.isExecuting ? 'border-indigo-500 ring-2 ring-indigo-200 shadow-indigo-100' : 'border-indigo-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-indigo-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-indigo-100 flex items-center justify-center text-indigo-600 font-bold text-xs">FE</div>
                    <div className="font-bold text-sm text-zinc-900">Feature Engineering</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-indigo-100 text-indigo-700 border-indigo-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            <div className="space-y-3 mb-4">
                <label className="text-[10px] text-zinc-500 font-semibold uppercase block">Select Script</label>
                <select
                    className="w-full text-xs border border-zinc-200 rounded p-1"
                    value={selectedScript}
                    onChange={handleScriptChange}
                >
                    <option value="">-- Choose Script --</option>
                    {scripts.map(s => (
                        <option key={s} value={s}>{s}</option>
                    ))}
                </select>
            </div>

            <div className="space-y-2">
                <div className="text-[10px] text-zinc-500 uppercase font-semibold flex justify-between items-center">
                    <span>Transformed Dataset</span>
                    <button
                        onClick={() => setShowExpanded(true)}
                        className="text-xs bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded hover:bg-indigo-100 transition-colors"
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

            <Handle type="source" position={Position.Right} className="!bg-indigo-500 !w-3 !h-3" />

            {/* Expanded Modal */}
            {showExpanded && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setShowExpanded(false)}>
                    <div className="bg-white rounded-xl shadow-2xl w-[90vw] h-[80vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-zinc-100 flex justify-between items-center bg-zinc-50">
                            <h3 className="font-bold text-lg text-zinc-800">Transformed Dataset View {loadingFullData && <span className="text-xs font-normal text-zinc-500 ml-2">(Loading...)</span>}</h3>
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

const formatTime = (ts: any) => {
    if (!ts) return '';
    try {
        if (!isNaN(ts)) {
            return new Date(Number(ts) * 1000).toLocaleTimeString();
        }
        return new Date(ts).toLocaleTimeString();
    } catch { return ts; }
}

export default memo(FeatureEngineeringNode);
