use algorithm::data_structure::graph::*;
use assembling::searching::banks::data_structure::int_graph::INodeData;
use assembling::searching::banks::data_structure::int_graph::IEdgeData;
use fnv::FnvHashMap;
use assembling::searching::banks::data_structure::int_graph::IntNode;
use std::cmp::Ordering;
use std::f32::INFINITY;

#[derive(Clone, Debug)]
pub struct TraceBack {
    edge_id: usize,
    parent_id: usize,
}

#[derive(Clone)]
pub struct ForwardPath {
    pub node_id: usize,
    pub outgoing_edge_id: usize,
}


pub struct DijkstraSSSPIterator<'a> {
    weighted_graph: &'a Graph<INodeData, IEdgeData>,
    distance: FnvHashMap<usize, f32>,
    pub source_id: usize,
    trace_back: FnvHashMap<usize, Vec<TraceBack>>,
    priority_queue: Vec<&'a IntNode>,
}

impl<'a> DijkstraSSSPIterator<'a> {

    pub fn new(weighted_graph: &'a Graph<INodeData, IEdgeData>, source_id: usize) -> DijkstraSSSPIterator<'a> {
        let source = weighted_graph.get_node_by_id(source_id);

        let mut distance: FnvHashMap<usize, f32> = Default::default();

        // create priority queue of all nodes we can reach from source_id including itself
        let mut priority_queue = Vec::with_capacity(weighted_graph.get_n_data_nodes());
        let mut trace_back: FnvHashMap<usize, Vec<TraceBack>> = Default::default();

        foreach_descendant_nodes(&weighted_graph, &source, |n: &IntNode| {
            distance.insert(n.id, INFINITY);
            priority_queue.push(n);
            trace_back.insert(n.id, Vec::new());
        });

        distance.insert(source_id, 0.0);
        trace_back.insert(source_id, Vec::new());
        // add source later, so we don't have to sort it        
        priority_queue.push(source);

        DijkstraSSSPIterator {
            weighted_graph,
            distance,
            source_id,
            priority_queue,
            trace_back
        }
    }

    #[inline]
    pub fn distance2nextnode(&self) -> f32 {
        self.distance[&self.priority_queue[self.priority_queue.len() - 1].id]
    }

    pub fn is_finish(&self) -> bool {
        self.priority_queue.len() == 0
    }

    pub fn travel_next_node(&mut self) -> usize {
        let u = self.priority_queue.pop().unwrap();

        if u.id == self.source_id {
            // u is source node, cannot apply heuristic
            for e in u.iter_outgoing_edges(&self.weighted_graph) {
                let v = e.get_target_node(&self.weighted_graph);
                let v_dist = self.distance[&u.id] + e.data.weight;

                if v_dist < self.distance[&v.id] {
                    // still check here, because may be we have multiple edges between two nodes
                    self.distance.insert(v.id, v_dist);
                    self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                } else {
                    self.trace_back.get_mut(&v.id).unwrap().push(TraceBack {edge_id: e.id, parent_id: u.id});
                }
            }
        } else {
            let u_parent_edge_id = self.trace_back[&u.id][0].edge_id;
            // can apply heuristic (mohsen heuristic), need to apply here, not in search because it doesn't
            // hold total order property
            for e in u.iter_outgoing_edges(&self.weighted_graph) {
                let v = e.get_target_node(&self.weighted_graph);
                let v_dist = self.distance[&u.id] + e.data.weight;

                if self.trace_back[&v.id].len() == 0 {
                    // haven't seen it before
                    self.distance.insert(v.id, v_dist);
                    self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                } else {
                    if e.data.overlaps_tag(&self.weighted_graph.get_edge_by_id(u_parent_edge_id).data) {
                        // this edge from same model with u.incoming_edge, what are we going to do?
                        // we want to give it high priority, but note that it's previous trace
                        // may also come from same model...
                        if v_dist < self.distance[&v.id] {
                            // at least this one has smaller weight, so it's obviously better
                            self.distance.insert(v.id, v_dist);
                            self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                        } else {
                            // now if the distance is greater or equal, does it trace also come from same model?
                            let v_curr_trace = self.trace_back[&v.id][0].clone();
                            if self.trace_back[&v_curr_trace.parent_id].len() == 0 {
                                // it doesn't have any parent, hmm, easy, it doesn't come from same model
                                // we give this edge higher priority, despite of the weight
                                self.distance.insert(v.id, v_dist);
                                self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                            } else {
                                // it does have parent, let's see if its parent comes from same model
                                let v_parent_curr_trace = self.trace_back[&v_curr_trace.parent_id][0].clone();
                                if self.weighted_graph.get_edge_by_id(v_parent_curr_trace.edge_id).data.overlaps_tag(&self.weighted_graph.get_edge_by_id(v_curr_trace.edge_id).data) {
                                    // it does come from same model
                                    if self.distance[&v.id] == v_dist {
                                        // same weight, we add trace_back to include this path as well
                                        self.trace_back.get_mut(&v.id).unwrap().push(TraceBack {edge_id: e.id, parent_id: u.id});
                                    }
                                    // it has less weight, we don't need to include in the trace_back
                                } else {
                                    // doesn't come from same model, so we choose e
                                    self.distance.insert(v.id, v_dist);
                                    self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                                }
                            }
                        }
                    } else {
                        // this edge is not from same model with u.incoming_edge, does trace of v come from same model?
                        let v_curr_trace = self.trace_back[&v.id][0].clone();
                        if self.trace_back[&v_curr_trace.parent_id].len() == 0 {
                            // no it's not, who has smaller weight will win!
                            if v_dist < self.distance[&v.id] {
                                self.distance.insert(v.id, v_dist);
                                self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                            } else if v_dist == self.distance[&v.id] {
                                self.trace_back.get_mut(&v.id).unwrap().push(TraceBack {edge_id: e.id, parent_id: u.id});
                            }
                        } else {
                            // it does have parent, let's see if its parent comes from same model
                            let v_parent_curr_trace = self.trace_back[&v_curr_trace.parent_id][0].clone();
                            if !self.weighted_graph.get_edge_by_id(v_parent_curr_trace.edge_id).data.overlaps_tag(&self.weighted_graph.get_edge_by_id(v_curr_trace.edge_id).data) {
                                // not same model, who has smaller weight will win!
                                if v_dist < self.distance[&v.id] {
                                    self.distance.insert(v.id, v_dist);
                                    self.trace_back.insert(v.id, vec![TraceBack {edge_id: e.id, parent_id: u.id}]);
                                } else if v_dist == self.distance[&v.id] {
                                    self.trace_back.get_mut(&v.id).unwrap().push(TraceBack {edge_id: e.id, parent_id: u.id});
                                }
                            }
                            // else, it does come from same model, so it has higher priority
                        }
                    }
                }
            }
        }

        let priority_queue = &mut self.priority_queue as *mut Vec<&'a IntNode>;
        unsafe {
            (&mut *priority_queue).sort_by(|a, b| self.distance[&b.id].partial_cmp(&self.distance[&a.id]).unwrap());
        }

        return u.id;
    }

    pub fn get_shortest_path(&self, mut target_id: usize) -> Vec<ForwardPath> {
        if !self.trace_back.contains_key(&target_id) {
            panic!("No path from source: {} to target: {}", self.source_id, target_id);
        }

        let mut path = Vec::new();
        while self.trace_back[&target_id].len() > 0 {
            let trace = &self.trace_back[&target_id][0];
            path.push(ForwardPath { node_id: trace.parent_id, outgoing_edge_id: trace.edge_id });
            target_id = trace.parent_id;
        }

        path.reverse();
        path
    }

    pub fn get_shortest_cost(&self, target_id: usize) -> f32 {
        self.distance[&target_id]
    }
}