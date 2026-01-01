import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';
import { getModels, getModelMetadata, getTrainingScripts } from '../../utils/api';

const TrainNode = ({ data, id }: NodeProps) => {
    const [scripts, setScripts] = useState<string[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const { setNodes } = useReactFlow();

    useEffect(() => {
        // [NEW] Fetch scripts instead of models
        getTrainingScripts()
            .then(res => setScripts(res?.scripts || []))
            .catch(err => {
                console.error("Failed to fetch training scripts:", err);
                setScripts([]);
            });
    }, []);

    const toggleScript = (scriptName: string) => {
        const currentSelected = data.scriptNames || (data.scriptName ? [data.scriptName] : []);
        let newSelected: string[];

        if (currentSelected.includes(scriptName)) {
            newSelected = currentSelected.filter((s: string) => s !== scriptName);
        } else {
            newSelected = [...currentSelected, scriptName];
        }

        // Update Node
        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                return {
                    ...node,
                    data: {
                        ...node.data,
                        scriptNames: newSelected,
                        scriptName: newSelected[0] || ''
                    }
                };
            }
            return node;
        }));
    };

    const selectedScripts = data.scriptNames || (data.scriptName ? [data.scriptName] : []);

    // Extract status from data.latestData (which comes from main.py's process_node return value)
    let statusMessage = "Idle";
    let isSuccess = false;
    let isError = false;

    if (data.latestData) {
        if (data.latestData.status) {
            statusMessage = data.latestData.status;
            if (statusMessage.includes("Ran") || statusMessage.includes("Success")) isSuccess = true;
            if (statusMessage.includes("Error") || statusMessage.includes("Failed")) isError = true;
        }
    }

    // Override if actively executing
    if (data.isExecuting) {
        const preview = selectedScripts.join(", ");
        statusMessage = `Running ${preview}...`;
        if (preview.length > 20) {
            statusMessage = `Running ${selectedScripts.length} scripts...`;
        }
    }

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-80 transition-all duration-200 ${data.isExecuting ? 'border-purple-500 ring-2 ring-purple-200 shadow-purple-100' : 'border-zinc-200'
            }`}>
            <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-purple-100 flex items-center justify-center text-purple-600 font-bold text-xs">TR</div>
                    <div className="font-bold text-sm text-zinc-900">Train Script</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-purple-100 text-purple-700 border-purple-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            <div className={`mb-3 p-2 border rounded text-center ${isSuccess ? 'bg-green-50 border-green-200' : isError ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200'
                }`}>
                <div className={`text-[10px] font-bold uppercase tracking-wider ${isSuccess ? 'text-green-600' : isError ? 'text-red-600' : 'text-gray-500'
                    }`}>
                    Status
                </div>
                <div className={`font-bold text-sm ${isSuccess ? 'text-green-800' : isError ? 'text-red-800' : 'text-gray-700'
                    }`}>
                    {statusMessage}
                </div>

                {/* Error Details / Logs */}
                {isError && data.latestData?.details && (
                    <div className="mt-2 text-left bg-red-100 p-2 rounded overflow-x-auto max-h-32 text-[10px] font-mono text-red-900 whitespace-pre-wrap">
                        {data.latestData.details.map((res: any, idx: number) => (
                            !res.success && (
                                <div key={idx} className="mb-2 border-b border-red-200 last:border-0 pb-1">
                                    <div className="font-bold">{res.script} Failed:</div>
                                    <div>{res.output || res.msg}</div>
                                </div>
                            )
                        ))}
                    </div>
                )}
            </div>

            <div className="space-y-3 relative">
                <div>
                    <label className="block text-[10px] uppercase font-semibold text-zinc-500 mb-1">Target Scripts</label>

                    {/* Multi-Select Trigger */}
                    <div
                        onClick={() => setIsOpen(!isOpen)}
                        className="w-full text-xs border border-zinc-300 rounded p-2 bg-white cursor-pointer hover:border-purple-400 flex justify-between items-center"
                    >
                        <span className={selectedScripts.length === 0 ? "text-zinc-400" : "text-zinc-900"}>
                            {selectedScripts.length === 0
                                ? "Select scripts..."
                                : `${selectedScripts.length} script(s) selected`}
                        </span>
                        <span className="text-zinc-400 text-[10px]">▼</span>
                    </div>

                    {/* Dropdown Menu */}
                    {isOpen && (
                        <div className="absolute top-16 left-0 w-full z-50 bg-white border border-zinc-200 shadow-xl rounded-md max-h-48 overflow-y-auto">
                            {(scripts || []).map(s => (
                                <div
                                    key={s}
                                    className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-50 cursor-pointer border-b border-zinc-50 last:border-0"
                                    onClick={() => toggleScript(s)}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedScripts.includes(s)}
                                        readOnly
                                        className="rounded text-purple-600 focus:ring-purple-500 w-3 h-3"
                                    />
                                    <span className="text-xs text-zinc-800">{s}</span>
                                </div>
                            ))}
                            {scripts.length === 0 && (
                                <div className="p-3 text-xs text-zinc-400 text-center">No scripts found</div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            <Handle type="target" position={Position.Left} className="!bg-zinc-400 !w-3 !h-3" />
            <Handle type="source" position={Position.Right} className="!bg-purple-500 !w-3 !h-3" />

            {/* Click outside closer could be added here or via proper dropdown hook, but simple toggle usually fine for node */}
            {isOpen && <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />}
        </div>
    );
};

export default memo(TrainNode);
