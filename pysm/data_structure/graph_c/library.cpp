//
//

#include <iostream>
#include "library.h"
#include <string>
#include <stdexcept>

using namespace std;

namespace isi {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Node

    Node::Node(int id, NodeType type, string lbl) {
        this->id = id;
        this->type = type;
        this->label = lbl;
        this->incoming_link_ids.reserve(2);
        this->outgoing_link_ids.reserve(12);
    }

    bool Node::equal(Node& that) {
        bool is_equal = this->id == that.id
               && this->type == that.type
               && this->label.compare(that.label) == 0
               && this->incoming_link_ids.size() == that.incoming_link_ids.size()
               && this->outgoing_link_ids.size() == that.outgoing_link_ids.size();

        if (!is_equal) {
            return false;
        }

        for (int i = this->incoming_link_ids.size() - 1; i >= 0; i--) {
            if (this->incoming_link_ids[i] != that.incoming_link_ids[i]) {
                return false;
            }
        }

        for (int i = this->outgoing_link_ids.size() - 1; i >= 0; i--) {
            if (this->outgoing_link_ids[i] != that.outgoing_link_ids[i]) {
                return false;
            }
        }

        return true;
    }

    void Node::add_incoming_link(int link_id) {
        this->incoming_link_ids.push_back(link_id);
    }

    void Node::add_outgoing_link(int link_id) {
        this->outgoing_link_ids.push_back(link_id);
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Link

    Link::Link(int id, LinkType type, string lbl, int source_id, int target_id) {
        this->id = id;
        this->type = type;
        this->label = std::move(lbl);
        this->source_id = source_id;
        this->target_id = target_id;
    }

    bool Link::equal(Link& that) {
        return this->id == that.id
              && this->type == that.type
              && this->label.compare(that.label) == 0
              && this->source_id == that.source_id
              && this->target_id == that.target_id;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Graph

    Graph::Graph(bool index_node_type, bool index_node_label, bool index_link_label, unsigned long estimated_n_nodes,
                 unsigned long estimated_n_links) {
        this->index_node_type = index_node_type;
        this->index_node_label = index_node_label;
        this->index_link_label = index_link_label;

        this->n_nodes = 0;
        this->n_links = 0;

        this->nodes.reserve(estimated_n_nodes);
        this->links.reserve(estimated_n_links);

        if (index_node_type) {
            this->class_node_index = new vector<int>();
            this->class_node_index->reserve(estimated_n_nodes);
            this->data_node_index = new vector<int>();
            this->data_node_index->reserve(estimated_n_links);
        } else {
            this->class_node_index = nullptr;
            this->data_node_index = nullptr;
        }

        if (index_node_label) {
            this->node_index_label = new google::dense_hash_map<string, vector<int>>();
            this->node_index_label->set_empty_key("empty-key-afedc06d-ab5f-45a2-aa11-073229c26f25");
            this->node_index_label->set_deleted_key("delete-key-e191ad73-0d11-45df-abf4-748067deff8e");
        } else {
            this->node_index_label = nullptr;
        }

        if (index_link_label) {
            this->link_index_label = new google::dense_hash_map<string, vector<int>>();
            this->link_index_label->set_empty_key("empty-key-afedc06d-ab5f-45a2-aa11-073229c26f25");
            this->link_index_label->set_deleted_key("delete-key-e191ad73-0d11-45df-abf4-748067deff8e");
        } else {
            this->link_index_label = nullptr;
        }
    }

    Graph::~Graph() {
        if (this->class_node_index != nullptr) {
            delete this->class_node_index;
            delete this->data_node_index;
        }
        if (this->node_index_label != nullptr) {
            delete this->node_index_label;
        }
        if (this->link_index_label != nullptr) {
            delete this->link_index_label;
        }
    }

    Node *Graph::add_new_node(NodeType type, string lbl) {
        Node n(this->n_nodes, type, std::move(lbl));
        this->add_node(n);
        return &this->nodes[this->n_nodes - 1];
    }

    Link *Graph::add_new_link(LinkType type, string lbl, int source_id, int target_id) {
        if (!this->has_node_with_id(source_id) or !this->has_node_with_id(target_id)) {
            throw std::invalid_argument("Graph->add_new_link func: cannot add link because source_id or target_id aren't present in the graph");
        }

        Link e(this->n_links, type, std::move(lbl), source_id, target_id);
        this->add_link(e);
        return &this->links[this->n_links - 1];
    }

    void Graph::add_node(Node& node) {
        this->nodes.push_back(node);
        this->update_node_index(node);
        this->n_nodes++;
    }

    void Graph::add_link(Link& link) {
        this->links.push_back(link);
        this->nodes[link.source_id].add_outgoing_link(link.id);
        this->nodes[link.target_id].add_incoming_link(link.id);
        this->update_link_index(link);
        this->n_links++;
    }

    void Graph::update_node_index(Node& node) {
        if (this->index_node_type) {
            if (node.type == CLASS_NODE) {
                this->class_node_index->push_back(node.id);
            } else {
                this->data_node_index->push_back(node.id);
            }
        }

        if (this->index_node_label) {
            if (this->node_index_label->find(node.label) == this->node_index_label->end()) {
                (*this->node_index_label)[node.label];
            }

            (*this->node_index_label)[node.label].push_back(node.id);
        }
    }

    void Graph::update_link_index(Link& link) {
        if (this->index_link_label) {
            if (this->link_index_label->find(link.label) == this->link_index_label->end()) {
                (*this->link_index_label)[link.label];
            }

            (*this->link_index_label)[link.label].push_back(link.id);
        }
    }

    bool Graph::has_node_with_id(int id) {
        return 0 <= id && id < this->n_nodes;
    }

    bool Graph::has_link_with_id(int id) {
        return 0 <= id && id < this->n_links;
    }

    Node *Graph::get_node_by_id(int id) {
        return &this->nodes[id];
    }

    Link *Graph::get_link_by_id(int id) {
        return &this->links[id];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Test

//    void print_node(Node& n) {
//        printf("(%d, %d, %s)\n", n.id, n.type, n.label.c_str());
//    }
//
//    void test() {
//        Node a0(0, CLASS_NODE, "crm:E39_Actor");
//        Node a1(1, CLASS_NODE, "crm:E42_Identifier");
//        Node a2(2, CLASS_NODE, "crm:E55_Type");
//        Node a3(3, CLASS_NODE, "crm:E63_Beginning_of_Existence");
//        Node a4(4, CLASS_NODE, "crm:E64_End_of_Existence");
//        Node a5(5, CLASS_NODE, "crm:E74_Group");
//        Node a6(6, CLASS_NODE, "crm:E82_Actor_Appellation");
//        Node a7(7, CLASS_NODE, "crm:E55_Type");
//        Node a8(8, CLASS_NODE, "crm:E52_Time-Span");
//        Node a9(9, CLASS_NODE, "crm:E53_Place");
//        Node a10(10, CLASS_NODE, "crm:E52_Time-Span");
//        Node a11(11, CLASS_NODE, "crm:E53_Place");
//        Node a12(12, DATA_NODE, "HN167985:BirthBeginDate");
//        Node a13(13, DATA_NODE, "HN163307:deathdate_uri");
//        Node a14(14, DATA_NODE, "HN116843:Values");
//        Node a15(15, DATA_NODE, "HN152247:artist_name_uri");
//        Node a16(16, DATA_NODE, "HN116828:Values");
//        Node a17(17, DATA_NODE, "HN169737:IdLabel");
//        Node a18(18, DATA_NODE, "HN116864:Values");
//        Node a19(19, DATA_NODE, "HN116822:Values");
//        Node a20(20, DATA_NODE, "HN157485:birthdate_uri");
//        Node a21(21, DATA_NODE, "HN169153:PrefIdURI");
//        Node a22(22, DATA_NODE, "HN116825:Values");
//        Node a23(23, DATA_NODE, "HN164478:begin_existence1_uri");
//        Node a24(24, DATA_NODE, "HN166816:DeathBeginDate");
//        Node a25(25, DATA_NODE, "HN168569:BirthEndDate");
//        Node a26(26, DATA_NODE, "HN158653:deathplace_uri");
//        Node a27(27, DATA_NODE, "HN166232:GenderURI_in_use");
//        Node a28(28, DATA_NODE, "HN152831:nationality_uri");
//        Node a29(29, DATA_NODE, "HN153415:row_uri");
//        Node a30(30, DATA_NODE, "HN116858:Values");
//        Node a31(31, DATA_NODE, "HN163894:End_Existence1_uri");
//        Node a32(32, DATA_NODE, "HN116852:Values");
//        Node a33(33, DATA_NODE, "HN167400:DeathEndDate");
//        Node a34(34, DATA_NODE, "HN116837:Values");
//        Node a35(35, DATA_NODE, "HN159237:GenderTypeURI");
//        Node a36(36, DATA_NODE, "HN165062:name_duplicate");
//        Node a37(37, DATA_NODE, "HN158069:birthplace_uri");
//        Node a38(38, DATA_NODE, "http://vocab.getty.edu/aat/300404670");
//        Node a39(39, DATA_NODE, "http://vocab.getty.edu/aat/300055147");
//        Node a40(40, DATA_NODE, "http://vocab.getty.edu/aat/300379842");
//        Node a41(41, DATA_NODE, "http://vocab.getty.edu/aat/300404670");
//        Node nodes[42] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41};
//
//        Link e0(0, OBJECT_PROPERTY, "crm:P1_is_identified_by",  0, 1);
//        Link e1(1, OBJECT_PROPERTY, "crm:P2_has_type",  0, 2);
//        Link e2(2, OBJECT_PROPERTY, "crm:P92i_was_brought_into_existence_by",  0, 3);
//        Link e3(3, OBJECT_PROPERTY, "crm:P93i_was_taken_out_of_existence_by",  0, 4);
//        Link e4(4, OBJECT_PROPERTY, "crm:P17i_is_current_or_former_member_of",  0, 5);
//        Link e5(5, OBJECT_PROPERTY, "crm:P131_is_identified_by",  0, 6);
//        Link e6(6, OBJECT_PROPERTY, "crm:P2_has_type",  2, 7);
//        Link e7(7, OBJECT_PROPERTY, "crm:P4_has_time-span",  3, 8);
//        Link e8(8, OBJECT_PROPERTY, "crm:P7_took_place_at",  3, 9);
//        Link e9(9, OBJECT_PROPERTY, "crm:P4_has_time-span",  4, 10);
//        Link e10(10, OBJECT_PROPERTY, "crm:P7_took_place_at",  4, 11);
//        Link e11(11, DATA_PROPERTY, "crm:P82a_begin_of_the_begin",  8, 12);
//        Link e12(12, URI_PROPERTY, "karma:classLink",  10, 13);
//        Link e13(13, DATA_PROPERTY, "rdfs:label",  11, 14);
//        Link e14(14, URI_PROPERTY, "karma:classLink",  6, 15);
//        Link e15(15, DATA_PROPERTY, "rdf:value",  1, 16);
//        Link e16(16, DATA_PROPERTY, "rdfs:label",  1, 17);
//        Link e17(17, DATA_PROPERTY, "rdfs:label",  8, 18);
//        Link e18(18, DATA_PROPERTY, "rdfs:label",  5, 19);
//        Link e19(19, URI_PROPERTY, "karma:classLink",  8, 20);
//        Link e20(20, URI_PROPERTY, "karma:classLink",  1, 21);
//        Link e21(21, DATA_PROPERTY, "rdfs:label",  9, 22);
//        Link e22(22, URI_PROPERTY, "karma:classLink",  3, 23);
//        Link e23(23, DATA_PROPERTY, "crm:P82a_begin_of_the_begin",  10, 24);
//        Link e24(24, DATA_PROPERTY, "crm:P82b_end_of_the_end",  8, 25);
//        Link e25(25, URI_PROPERTY, "karma:classLink",  11, 26);
//        Link e26(26, URI_PROPERTY, "karma:classLink",  2, 27);
//        Link e27(27, URI_PROPERTY, "karma:classLink",  5, 28);
//        Link e28(28, URI_PROPERTY, "karma:classLink",  0, 29);
//        Link e29(29, DATA_PROPERTY, "rdfs:label",  10, 30);
//        Link e30(30, URI_PROPERTY, "karma:classLink",  4, 31);
//        Link e31(31, DATA_PROPERTY, "rdf:value",  6, 32);
//        Link e32(32, DATA_PROPERTY, "crm:P82b_end_of_the_end",  10, 33);
//        Link e33(33, DATA_PROPERTY, "rdfs:label",  2, 34);
//        Link e34(34, URI_PROPERTY, "karma:classLink",  7, 35);
//        Link e35(35, DATA_PROPERTY, "rdfs:label",  0, 36);
//        Link e36(36, URI_PROPERTY, "karma:classLink",  9, 37);
//        Link e37(37, OBJECT_PROPERTY, "crm:P2_has_type",  1, 38);
//        Link e38(38, OBJECT_PROPERTY, "crm:P2_has_type",  6, 41);
//        Link e39(39, OBJECT_PROPERTY, "crm:P2_has_type",  5, 40);
//        Link e40(40, OBJECT_PROPERTY, "skos:broadMatch",  7, 39);
//        Link links[41] = {e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36,e37,e38,e39,e40};
//
//        Graph *g = new Graph(true, true, true, 16, 32);
//
//        for (auto &n : nodes) {
//            g->add_new_node(n.type, n.label);
//        }
//        for (auto &e : links) {
//            g->add_new_link(e.type, e.label, e.source_id, e.target_id);
//        }
//
//        Node *ptr;
//        cout << "=============================" << g->class_node_index->size() << endl;
//
////        for (int i = 0; i < g.n_nodes; i++) {
////            ptr = g.get_node_by_id(i);
////            print_node(*ptr);
////        }
//
//        for (unsigned int i = 0; i < g->class_node_index->size(); i++) {
//            ptr = g->get_node_by_id((*g->class_node_index)[i]);
//            print_node(*ptr);
//        }
//
//        cout << "=============================" << g->data_node_index->size() << endl;
//
//        for (unsigned int i = 0; i < g->data_node_index->size(); i++) {
//            ptr = g->get_node_by_id((*g->data_node_index)[i]);
//            print_node(*ptr);
//        }
//
//        cout << "=============================" << g->link_index_label->size() << endl;
//        vector<int> &idarray = (*g->link_index_label)["karma:classLink"];
//        for (unsigned int i = 0; i < idarray.size(); i++) {
//            ptr = g->get_node_by_id(idarray[i]);
//            print_node(*ptr);
//        }
//        delete g;
//    }
//
//    void test2() {
////        string nodeNames[4] = {"Joah", "John", "Joah", "Peter"};
////        NodeType nodeTypes[4] = {CLASS_NODE, CLASS_NODE, CLASS_NODE, DATA_NODE};
////        int sourceIds[3] = {0, 2, 1};
////        int targetIds[3] = {1, 3, 10};
////        Graph *g = new Graph(true, true, true, 24, 23);
////
////        for (int i = 0; i < 80; i++) {
////            g->add_new_node(nodeTypes[i % 4], nodeNames[i % 4]);
////        }
////
////        for (int i = 0; i < 60; i++) {
////            g->add_new_link(URI_PROPERTY, "Friend", sourceIds[i%3], targetIds[i%3]);
////        }
////        delete g;
//
//        Graph *g0 = new Graph(false, false, false, 24, 23);
//        Node *a = g0->add_new_node(CLASS_NODE, "Joan");
////        a->incoming_link_ids.push_back(5);
//        printf("(%d)\n", a->incoming_link_ids.size());
//        Graph *g1 = new Graph(false, false, false, 24, 23);
//        Node *b = g1->add_new_node(CLASS_NODE, "Joan");
//        cout << a->equal(*b) << endl;
//    }
}

//int main() {
//    isi::test2();
//    cout << ">>> TERMINATE" << endl;
//}