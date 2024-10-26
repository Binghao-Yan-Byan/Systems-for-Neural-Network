#pragma once
#include <string>
#include <iostream>
using std::string;
using std::cout;
using std::endl;
typedef uint32_t vid_t;

class graph_t {
public:
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* dgrs) {
        this->a_vcount = (int32_t)a_vcount;
        this->a_dstsize = (int32_t)a_dstsize;
        this->offset = (int32_t*)a_offset;
        this->nebrs = (int32_t*)a_nebrs;
        this->dgrs = (int32_t*)dgrs;
    };
    void save_graph(const string& full_path) {};
    void load_graph(const string& full_path) {};
    void print_graph(){
        cout << "OFFSET" << endl;
        for(int i = 0; i < this->a_vcount; i ++){
            cout<<this->offset[i]<<" ";
        }
        cout << "NEIGHBORS" << endl;
        for(int i = 0; i < this->a_dstsize; i ++){
            cout<<this->nebrs[i]<<" ";
        }
        cout << "DEGREES" << endl;
        for(int i = 0; i < this->a_vcount; i ++){
            cout<<this->dgrs[i]<<" ";
        }
    };
    void load_graph_noeid(const string& full_path) {};
    int get_vcount() {return this->a_vcount;};
    int get_ecount() {return this->a_dstsize;};

    int32_t a_vcount;
    int32_t a_dstsize;
    int32_t *offset, *nebrs, *dgrs;
};
