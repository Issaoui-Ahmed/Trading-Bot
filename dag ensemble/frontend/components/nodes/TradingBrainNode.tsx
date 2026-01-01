
import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';

const TradingBrainNode = ({ data, id }: NodeProps) => {
    const { setNodes } = useReactFlow();
    const [scripts, setScripts] = useState<string[]>([]);
    const [selectedScript, setSelectedScript] = useState(data.scriptName || '');

    // Fetch available scripts on mount
    useEffect(() => {
        fetch('http://localhost:8001/trading-brains')
            .then(res => res.json())
            .then(data => {
                setScripts(data.scripts || []);
            })
            .catch(err => console.error("Failed to fetch trading brains", err));
    }, []);

    const handleScriptChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newVal = e.target.value;
        setSelectedScript(newVal);

        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, scriptName: newVal } };
                }
                return node;
            })
        );
    };

    const result = data.latestData; // This will come from backend execution of the brain

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-64 transition-all duration-200 ${data.isExecuting ? 'border-orange-500 ring-2 ring-orange-200' : 'border-orange-200'}`}>
            <Handle type="target" position={Position.Left} className="!bg-orange-500 !w-3 !h-3" />

            <div className="flex items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-orange-100 flex items-center justify-center text-orange-600 font-bold text-xs">TB</div>
                    <div className="font-bold text-sm text-zinc-900">Trading Brain</div>
                </div>
                {/* Execution Badge */}
                {data.isExecuting && (
                    <div className="w-2 h-2 rounded-full bg-orange-500 animate-ping" />
                )}
            </div>

            <div className="space-y-3 mb-4">
                <label className="text-[10px] text-zinc-500 font-semibold uppercase block">Select Brain Script</label>
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

            <div className="bg-zinc-50 border border-zinc-100 rounded p-2 text-xs">
                <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1">Latest Decision</div>
                {result ? (
                    <div className="font-mono text-zinc-700">
                        {result.error ? (
                            <div className="text-red-500 font-semibold break-words">{result.error}</div>
                        ) : (
                            <>
                                <div className={`font-bold ${result.action === 'buy' ? 'text-green-600' : result.action === 'sell' ? 'text-red-600' : 'text-zinc-500'}`}>
                                    ACTION: {result.action ? result.action.toUpperCase() : 'N/A'}
                                </div>
                                <div>VOL: {result.volume}</div>
                            </>
                        )}
                    </div>
                ) : (
                    <div className="text-zinc-400 italic">No decision yet...</div>
                )}
            </div>

            <Handle type="source" position={Position.Right} className="!bg-orange-500 !w-3 !h-3" />
        </div>
    );
};

export default memo(TradingBrainNode);
