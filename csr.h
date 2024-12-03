#pragma once
#include <string>
#include <iostream>
#include <cuda_runtime.h>
using std::string;
using std::cout;
using std::endl;
typedef uint32_t vid_t;

class graph_t {
public:
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* dgrs) {
        this->vcount = (int32_t)a_vcount;
        this->ecount = (int32_t)a_dstsize;
        
        cudaMalloc((void**)&this->nebrs, ecount*sizeof(int32_t));
        cudaMemcpy(this->nebrs, a_nebrs, ecount*sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&this->offset, (vcount+1)*sizeof(int32_t));
        cudaMemcpy(this->offset, a_offset, (vcount+1)*sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&this->dgrs, vcount*sizeof(int32_t));
        cudaMemcpy(this->dgrs, dgrs, vcount*sizeof(int32_t), cudaMemcpyHostToDevice);
    };
    void save_graph(const string& full_path) {};
    void load_graph(const string& full_path) {};
    void print_graph(){
        for(int i = 0; i < this->vcount+1; i ++){
            cout<<this->offset[i]<<" ";
        }
        cout << endl;
        for(int i = 0; i < this->ecount; i ++){
            cout<<this->nebrs[i]<<" ";
        }
        cout << endl;
        for(int i = 0; i < this->vcount; i ++){
            cout<<this->dgrs[i]<<" ";
        }
        cout << endl;
    };
    void load_graph_noeid(const string& full_path) {};
    int get_vcount() {return this->vcount;};
    int get_ecount() {return this->ecount;};
    ~graph_t(){
        cudaFree(offset);
        cudaFree(nebrs);
        cudaFree(dgrs);
    }
    int32_t vcount;
    int32_t ecount;
    int32_t *offset, *nebrs, *dgrs;
};
