use assembling::searching::beam_search::data_structure::ExploringNodes;
use assembling::searching::beam_search::data_structure::SearchNodeExtraData;
use assembling::searching::beam_search::data_structure::SearchNode;
use assembling::searching::beam_search::data_structure::SearchArgs;

#[inline]
pub fn no_stop<'a, T: 'a + SearchNodeExtraData, A>(_args: &mut SearchArgs<'a, T, A>, _iter_no: usize, _search_nodes: &SearchNode<T>) -> bool {
    false
}