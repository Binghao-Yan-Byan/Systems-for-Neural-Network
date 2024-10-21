inline void export_kernel(py::module &m) { 
    m.def("gemm",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gemm(input1_array, input2_array, output_array);
    }
  );
    m.def("spmm",[](py::capsule& input1, py::capsule& input2, py::capsule& input3, py::capsule& input4, py::capsule& output){
        array1d_t<float> input1_array = capsule_to_array1d(input1);
        array1d_t<float> input2_array = capsule_to_array1d(input2);
        array1d_t<float> input3_array = capsule_to_array1d(input3);
        array2d_t<float> input4_array = capsule_to_array2d(input4);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return spmm(input1_array, input2_array, input3_array, input4_array, output_array);
    }
  );
}