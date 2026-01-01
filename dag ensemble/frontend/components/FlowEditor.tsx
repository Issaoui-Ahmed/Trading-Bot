'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import ReactFlow, {
    addEdge,
    Background,
    Controls,
    Connection,
    Edge,
    Node,
    useNodesState,
    useEdgesState,
    ReactFlowProvider,
    ReactFlowInstance,
    BackgroundVariant
} from 'reactflow';
import 'reactflow/dist/style.css';

import ModelNode from './nodes/ModelNode';
import LiveDataFeedNode from './nodes/LiveDataFeedNode';
import FeatureEngineeringNode from './nodes/FeatureEngineeringNode';
import MergeNode from './nodes/MergeNode';
import TradingBrainNode from './nodes/TradingBrainNode';
import BrokerNode from './nodes/BrokerNode';
import EvalNode from './nodes/EvalNode';
import DataReplayerNode from './nodes/DataReplayerNode';
import RandomActionNode from './nodes/RandomActionNode';
import IfElseNode from './nodes/IfElseNode';
import DataStoreNode from './nodes/DataStoreNode';
import ReadNode from './nodes/ReadNode';
import TrainNode from './nodes/TrainNode';
import Sidebar from './Sidebar';
import { updateWorkflow, getStreamData, pauseWorkflow, resumeWorkflow, resetWorkflow } from '../utils/api';

const nodeTypes = {
    model: ModelNode,
    trainNode: TrainNode,
    liveDataFeed: LiveDataFeedNode,
    featureEngineering: FeatureEngineeringNode,
    mergeNode: MergeNode,
    tradingBrain: TradingBrainNode,
    brokerNode: BrokerNode,
    evalNode: EvalNode,
    dataReplayer: DataReplayerNode,
    randomAction: RandomActionNode,
    ifElseNode: IfElseNode,
    dataStore: DataStoreNode,
    readNode: ReadNode,
};

const getId = () => `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const FlowEditorContent = () => {
    const reactFlowWrapper = useRef<HTMLDivElement>(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
    const [workflowTotalTime, setWorkflowTotalTime] = useState<number | null>(null);
    const [workflowInterval, setWorkflowInterval] = useState<number | null>(null);
    const [isPaused, setIsPaused] = useState<boolean>(false);

    // Refs for accessing latest state in timers
    const nodesRef = useRef(nodes);
    const edgesRef = useRef(edges);

    useEffect(() => { nodesRef.current = nodes; }, [nodes]);
    useEffect(() => { edgesRef.current = edges; }, [edges]);

    // --- Stream Polling ---
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const data = await getStreamData();
                // data = { inputs, results, feed_snapshot, active_feeds, execution_stats }

                if (data.execution_stats) {
                    setWorkflowTotalTime(data.execution_stats.workflow_total_time_ms);
                    setWorkflowInterval(data.execution_stats.workflow_interval_ms);
                }

                if (data.workflow_status) {
                    setIsPaused(data.workflow_status.is_paused);
                }

                setNodes(currentNodes => currentNodes.map(node => {
                    let newData: any = {};

                    // 1. Handling Execution Stats
                    if (data.execution_stats && data.execution_stats.nodes && data.execution_stats.nodes[node.id]) {
                        const stats = data.execution_stats.nodes[node.id];
                        newData.executionTime = stats.last_execution_time_ms;
                        newData.isExecuting = stats.status === 'executing';
                    } else {
                        // Default if not present in stats yet
                        newData.isExecuting = false;
                    }

                    // 2. Handling Node Specific Data
                    if (node.type === 'liveDataFeed' || node.type === 'dataReplayer') {
                        // Backend sends LATEST_DATA as 'inputs'.
                        // Backend sends LATEST_DATA as 'inputs'.
                        // Check for node-specific data first
                        let nodeData = data.inputs[node.id];

                        // Fallback to global flattened data if specific is missing
                        if (!nodeData && data.inputs.open !== undefined) {
                            nodeData = data.inputs;
                        }

                        if (nodeData) {
                            newData.latestData = nodeData;

                            // Inject active feeds from node-specific data if available
                            if ((nodeData as any).activeFeeds) {
                                newData.activeFeeds = (nodeData as any).activeFeeds;
                            }
                        }

                        // Inject snapshot if available
                        if (data.feed_snapshot) {
                            newData.feedSnapshot = data.feed_snapshot;
                        }

                        // Fallback to global active feeds if not found in node data
                        if (!newData.activeFeeds && data.active_feeds) {
                            newData.activeFeeds = data.active_feeds;
                        }


                        // Inject Replayer Stats
                        if (data.replayer_stats && data.replayer_stats[node.id]) {
                            newData.replayerStats = data.replayer_stats[node.id];
                        }

                    } else if (node.type === 'featureEngineering' || node.type === 'mergeNode' || node.type === 'dataStore') {

                        // Check for node-specific data in inputs
                        if (data.inputs && data.inputs[node.id]) {
                            newData.latestData = data.inputs[node.id];
                        }
                    } else if (node.type === 'model') {
                        // Check if we have a result for this node
                        if (data.results && data.results[node.id] !== undefined) {
                            newData.inferenceResult = data.results[node.id];
                        }
                    } else if (node.type === 'tradingBrain' || node.type === 'paperTrading' || node.type === 'brokerNode' || node.type === 'evalNode' || node.type === 'randomAction' || node.type === 'ifElseNode') {
                        if (data.inputs && data.inputs[node.id]) {
                            newData.latestData = data.inputs[node.id];
                        }
                    }

                    if (Object.keys(newData).length > 0) {
                        return { ...node, data: { ...node.data, ...newData } };
                    }
                    return node;
                }));

            } catch (e) {
                console.error("Stream polling error", e);
            }
        }, 1000); // Poll every 1s (Backend updates every 3s)

        return () => clearInterval(interval);
    }, [setNodes]);

    // --- Workflow Update ---
    // Whenever nodes/edges change, send to backend
    useEffect(() => {
        if (nodes.length > 0) {
            // Debounce this in production, but simpler here
            const simplifiedNodes = nodes.map(n => ({
                id: n.id,
                type: n.type,
                data: n.data
            }));
            const simplifiedEdges = edges.map(e => ({
                source: e.source,
                target: e.target,
                sourceHandle: e.sourceHandle,
                targetHandle: e.targetHandle
            }));
            updateWorkflow(simplifiedNodes, simplifiedEdges).catch(e => console.error(e));
        }
    }, [nodes, edges]);


    const onConnect = useCallback((params: Connection) => {
        // Validation: Logic to restrict Many-to-One connections
        const targetNodeId = params.target;

        // Find the target node object to check its type
        const targetNode = nodesRef.current.find(n => n.id === targetNodeId);

        if (!targetNode) return; // Should not happen

        // Check if there are already edges connected to this target handle
        const existingEdges = edgesRef.current.filter(e => e.target === targetNodeId);

        // If target is NOT a MergeNode, and it already has an input, REJECT
        if (targetNode.type !== 'mergeNode' && existingEdges.length > 0) {
            alert("This node allows only one input. Use a Merge Node to combine multiple streams.");
            return;
        }

        setEdges((eds) => addEdge(params, eds));
    }, [setEdges, nodesRef, edgesRef]);

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();
            if (!reactFlowWrapper.current || !reactFlowInstance) return;

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) return;

            const position = reactFlowInstance.screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            const newNode: Node = {
                id: getId(),
                type,
                position,
                data: { label: `${type} node` },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance, setNodes]
    );

    return (
        <div className="flex w-full h-full">
            <Sidebar
                nodes={nodes}
                edges={edges}
                setNodes={setNodes}
                setEdges={setEdges}
            />
            <div className="flex-grow h-full relative" ref={reactFlowWrapper}>
                {(workflowTotalTime !== null || workflowInterval !== null) && (
                    <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur-sm p-3 rounded-lg shadow-md border border-gray-200 flex space-x-4">
                        {workflowTotalTime !== null && (
                            <div>
                                <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-1">Latency</div>
                                <div className="text-xl font-bold text-gray-800 font-mono">
                                    {workflowTotalTime.toFixed(2)}<span className="text-sm text-gray-500 ml-1">ms</span>
                                </div>
                            </div>
                        )}
                        {workflowInterval !== null && workflowInterval > 0 && (
                            <div className="pl-4 border-l border-gray-300">
                                <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-1">Frequency</div>
                                <div className="text-xl font-bold text-gray-800 font-mono">
                                    {(workflowInterval / 1000).toFixed(1)}<span className="text-sm text-gray-500 ml-1">s</span>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-white/90 backdrop-blur-sm p-1.5 rounded-lg shadow-md border border-gray-200 flex gap-2">
                    <button
                        className={`flex items-center gap-2 px-4 py-2 rounded text-xs font-bold transition-colors ${isPaused
                            ? 'bg-green-100 text-green-700 hover:bg-green-200 border border-green-200'
                            : 'bg-amber-100 text-amber-700 hover:bg-amber-200 border border-amber-200'
                            }`}
                        onClick={async () => {
                            if (isPaused) {
                                await resumeWorkflow();
                            } else {
                                await pauseWorkflow();
                            }
                            // Optimistic update
                            setIsPaused(!isPaused);
                        }}
                    >
                        {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
                    </button>
                    <div className="w-px bg-zinc-200 mx-1"></div>
                    <button
                        className="flex items-center gap-2 px-4 py-2 rounded text-xs font-bold bg-zinc-50 text-zinc-600 hover:bg-zinc-100 border border-zinc-200 transition-colors"
                        onClick={() => {
                            if (confirm('Are you sure you want to reset the entire workflow? This will clear all accumulated data and trading history.')) {
                                resetWorkflow();
                            }
                        }}
                    >
                        ↺ RESET ALL
                    </button>
                </div>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    fitView
                >
                    <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
                    <Controls />
                </ReactFlow>
            </div>
        </div>
    );
};

export default function FlowEditor() {
    return (
        <ReactFlowProvider>
            <FlowEditorContent />
        </ReactFlowProvider>
    );
}
