//
//

#ifndef GRAPH_LIBRARY_H
#define GRAPH_LIBRARY_H

#include <string>
#include <vector>
#include <sparsehash/dense_hash_map>

using namespace std;

namespace isi {
    enum NodeType {
        CLASS_NODE = 1,
        DATA_NODE = 2
    };

    enum LinkType {
        UNSPECIFIED = 0,
        UNKNOWN = 1,
        URI_PROPERTY = 2,
        OBJECT_PROPERTY = 3,
        DATA_PROPERTY = 4
    };

    class Node {
    public:
        int id;
        NodeType type;
        string label;

        vector<int> incoming_link_ids;
        vector<int> outgoing_link_ids;

        Node(int, NodeType, string);
        bool equal(Node&);
        void add_incoming_link(int);
        void add_outgoing_link(int);
    };

    class Link {
    public:
        int id, source_id, target_id;
        LinkType type;
        string label;

        Link(int, LinkType, string, int, int);
        bool equal(Link&);
    };

    class Graph {
    public:
        bool index_node_type, index_node_label, index_link_label;
        int n_nodes, n_links;

        vector<Node> nodes;
        vector<Link> links;
        vector<int> *class_node_index, *data_node_index;
        google::dense_hash_map<string, vector<int>> *node_index_label, *link_index_label;

        Graph(bool, bool, bool, unsigned long, unsigned long);
        ~Graph();

        Node* add_new_node(NodeType, string);
        Link* add_new_link(LinkType, string, int, int);
        // Those functions will actually call a copied constructor, so any change outside won't reflect into graph
        bool has_node_with_id(int);
        bool has_link_with_id(int);
        Node* get_node_by_id(int);
        Link* get_link_by_id(int);
    private:
        void add_node(Node&);
        void add_link(Link&);
        void update_node_index(Node&);
        void update_link_index(Link&);
    };

    void test();
}

#endif //GRAPH_LIBRARY_H
