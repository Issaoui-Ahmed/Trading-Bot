import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';
import { getModels, getModelMetadata } from '../../utils/api';

const ModelNode = ({ data, id }: NodeProps) => {
    const [models, setModels] = useState<string[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const { setNodes } = useReactFlow();

    useEffect(() => {
        getModels()
            .then(res => setModels(res?.models || []))
            .catch(err => {
                console.error("Failed to fetch models:", err);
                setModels([]);
            });
    }, []);

    const toggleModel = async (modelName: string) => {
        const currentSelected = data.modelNames || (data.modelName ? [data.modelName] : []);
        let newSelected: string[];

        if (currentSelected.includes(modelName)) {
            newSelected = currentSelected.filter((m: string) => m !== modelName);
        } else {
            newSelected = [...currentSelected, modelName];
        }

        // If only one selected, fetch requirements for it (legacy behavior)
        let requirements = null;
        if (newSelected.length === 1) {
            const meta = await getModelMetadata(newSelected[0]);
            if (meta && meta.training_context) {
                requirements = {
                    pair: meta.training_context.pair,
                    timeframe: meta.training_context.timeframe
                };
            }
        }

        // Update Node
        setNodes((nds) => nds.map((node) => {
            if (node.id === id) {
                return {
                    ...node,
                    data: {
                        ...node.data,
                        modelNames: newSelected,
                        modelName: newSelected[0] || '', // Fallback/Legacy
                        requirements
                    }
                };
            }
            return node;
        }));
    };

    const selectedModels = data.modelNames || (data.modelName ? [data.modelName] : []);

    // Render Logic for Inference Result
    const renderResult = () => {
        const result = data.inferenceResult;
        if (result === undefined || result === null) return null;

        // If Object (Multi-Model)
        if (typeof result === 'object' && result !== null) {
            return (
                <div className="text-xs text-left space-y-1">
                    {Object.entries(result).map(([name, val]) => (
                        <div key={name} className="flex justify-between border-b border-blue-100 last:border-0 pb-0.5">
                            <span className="font-semibold text-blue-700 truncate max-w-[100px]" title={name}>{name}</span>
                            <span className="font-mono text-zinc-700">
                                {typeof val === 'number' ? val.toFixed(4) : String(val)}
                            </span>
                        </div>
                    ))}
                </div>
            );
        }

        // Single Value (Legacy/Single)
        return (
            <div className={`font-bold ${!isNaN(Number(result)) ? 'text-xl text-blue-800' : 'text-xs text-red-700 break-words'}`}>
                {!isNaN(Number(result))
                    ? Number(result).toFixed(4)
                    : String(result)}
            </div>
        );
    };

    return (
        <div className={`p-4 border rounded-lg bg-white shadow-md w-80 transition-all duration-200 ${data.isExecuting ? 'border-blue-500 ring-2 ring-blue-200 shadow-blue-100' : 'border-zinc-200'
            }`}>
            <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-orange-100 flex items-center justify-center text-orange-600 font-bold text-xs">ML</div>
                    <div className="font-bold text-sm text-zinc-900">Predict</div>
                </div>
                {/* Execution Time Badge */}
                {data.executionTime !== undefined && (
                    <div className={`text-[10px] font-mono px-1.5 py-0.5 rounded border flex items-center gap-1 ${data.isExecuting ? 'bg-blue-100 text-blue-700 border-blue-200 animate-pulse' : 'bg-zinc-50 text-zinc-500 border-zinc-200'
                        }`}>
                        {data.isExecuting && <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-ping" />}
                        {data.executionTime.toFixed(2)}ms
                    </div>
                )}
            </div>

            {data.inferenceResult !== undefined && (
                <div className={`mb-3 p-2 border rounded text-center ${typeof data.inferenceResult === 'object' || !isNaN(Number(data.inferenceResult)) ? 'bg-blue-50 border-blue-200' : 'bg-red-50 border-red-200'
                    }`}>
                    <div className={`text-[10px] font-bold uppercase tracking-wider mb-1 ${typeof data.inferenceResult === 'object' || !isNaN(Number(data.inferenceResult)) ? 'text-blue-500' : 'text-red-500'
                        }`}>
                        {typeof data.inferenceResult === 'object' ? 'Predictions' : 'Inference Result'}
                    </div>
                    {renderResult()}
                </div>
            )}

            <div className="space-y-3 relative">
                <div>
                    <label className="block text-[10px] uppercase font-semibold text-zinc-500 mb-1">Select Models</label>

                    {/* Multi-Select Trigger */}
                    <div
                        onClick={() => setIsOpen(!isOpen)}
                        className="w-full text-xs border border-zinc-300 rounded p-2 bg-white cursor-pointer hover:border-blue-400 flex justify-between items-center"
                    >
                        <span className={selectedModels.length === 0 ? "text-zinc-400" : "text-zinc-900"}>
                            {selectedModels.length === 0
                                ? "Select models..."
                                : `${selectedModels.length} model(s) selected`}
                        </span>
                        <span className="text-zinc-400 text-[10px]">▼</span>
                    </div>

                    {/* Dropdown Menu */}
                    {isOpen && (
                        <div className="absolute top-16 left-0 w-full z-50 bg-white border border-zinc-200 shadow-xl rounded-md max-h-48 overflow-y-auto">
                            {(models || []).map(m => (
                                <div
                                    key={m}
                                    className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-50 cursor-pointer border-b border-zinc-50 last:border-0"
                                    onClick={() => toggleModel(m)}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedModels.includes(m)}
                                        readOnly
                                        className="rounded text-blue-600 focus:ring-blue-500 w-3 h-3"
                                    />
                                    <span className="text-xs text-zinc-800">{m}</span>
                                </div>
                            ))}
                            {models.length === 0 && (
                                <div className="p-3 text-xs text-zinc-400 text-center">No models found</div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            <Handle type="target" position={Position.Left} className="!bg-zinc-400 !w-3 !h-3" />
            <Handle type="source" position={Position.Right} className="!bg-blue-500 !w-3 !h-3" />

            {isOpen && <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />}
        </div>
    );
};

export default memo(ModelNode);
