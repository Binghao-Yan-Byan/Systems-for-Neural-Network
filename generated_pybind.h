inline void export_kernel(py::module &m) { 
    m.def("gemm",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gemm(input1_array, input2_array, output_array);
    }
  );
}