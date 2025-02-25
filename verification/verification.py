import onnx
from onnx import numpy_helper
import onnxruntime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import json
# import cv2
import argparse
import sys
import io
import cProfile
import pstats
import time
import os
sys.setrecursionlimit(500000)


class ModelInfo:
    def __init__(self, model_path, pipeline_type):
        self.onnx_model = onnx.load(model_path)
        self.pipeline_type = pipeline_type

    def load_weight(self):
        print("==================== Load Weight ====================")
        # load weight to OriginalWeightDict
        weights, names = [], []
        for t in self.onnx_model.graph.initializer:
            weights.append(numpy_helper.to_array(t))
            names.append(t.name)
        self.OriginalWeightDict = dict(zip(names, weights))
        for k, v in self.OriginalWeightDict.items():
            print(k, "  ", v.shape)

        # OutputToWeightDict : node_name → weight_name
        # OutputToBiasDict: node_name → bias_name
        # So node_index → node_name → weight_name → weight_data
        self.OutputToWeightDict = {}
        self.OutputToBiasDict = {}
        for node in self.onnx_model.graph.node:
            if node.op_type == "Conv" or node.op_type == "Gemm":
                print("DEBUG_loadweight")
                print(node.output)
                print(node.input)
                if len(node.input) == 2:
                    self.OutputToWeightDict[node.output[0]] = node.input[1]
                elif len(node.input) == 3:
                    self.OutputToWeightDict[node.output[0]] = node.input[1]
                    self.OutputToBiasDict[node.output[0]] = node.input[2]

        # for group_conv:
        self.group_weight_2_conv_name = {}
        for i, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "Conv":
                for attribute in node.attribute:
                    if attribute.name == "group" and attribute.i != 1:
                        self.group_weight_2_conv_name[node.input[1]] = node.output[0]

        # convert original weight (4-d) to weight matrix (2-d, height=K*K*Cin, Width=Cout)
        self.GEMMWeightDict = {}
        for k, v in self.OriginalWeightDict.items():
            if k in self.group_weight_2_conv_name: # Group CONV Weight
                self.GEMMWeightDict[k] = v.transpose((2,3,1,0))
            elif len(v.shape) == 4:  # CONV Weight
                self.GEMMWeightDict[k] = v.transpose( (0, 2, 3, 1)).reshape((v.shape[0], v[0].size)).transpose()
            else:  # FC Weight and Bias
                self.GEMMWeightDict[k] = v.transpose()

        # if self.pipeline_type == "element":
        #     # node_name = "vgg0_dense0_fwd"
        #     # if node_name in self.OutputToWeightDict:
        #     #     weight_name = self.OutputToWeightDict[node_name]
        #     #     weight_matrix = self.GEMMWeightDict[weight_name]
        #     #     weight_matrix = weight_matrix.reshape((512, 7, 7, 4096))
        #     #     weight_matrix = weight_matrix.transpose(1, 2, 0, 3)
        #     #     weight_shape = weight_matrix.shape
        #     #     self.GEMMWeightDict[weight_name] = weight_matrix.reshape(weight_matrix[:, :, :, 0].size, weight_shape[3])

    def eliminate_no_consider_OP(self):
        # eliminate LRN OP
        for i, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "LRN":
                self.onnx_model.graph.node[i].attribute[1].f = 1e-8  # "alpha"
                self.onnx_model.graph.node[i].attribute[2].f = 1e-8  # "beta"
                self.onnx_model.graph.node[i].attribute[3].f = 1e-8  # "bias"
            if node.op_type == "Clip":
                self.onnx_model.graph.node[i].attribute[1].f = -1e8
                self.onnx_model.graph.node[i].attribute[0].f = 1e8

        # for idx,tensor in enumerate(self.onnx_model.graph.initializer):
        #     if len(tensor.dims) == 1:
        #         # for float_data:
        #         if len(tensor.float_data) != 0:
        #             if not "batchnorm" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 0.0 #element by element
        #             elif "beta" in tensor.name or "mean" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 0.0
        #             elif "gamma" in tensor.name or "var" in tensor.name:
        #                 for l in range(len(tensor.float_data)):
        #                     self.onnx_model.graph.initializer[idx].float_data[l] = 1.0
        #         # for raw_data:
        #         else:
        #             raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
        #             raw_data_float32 = np.zeros(raw_data_float32.shape, dtype=np.float32)
        #             self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()

        BN_zero_params = []
        BN_one_params = []
        for node in self.onnx_model.graph.node:
            if node.op_type == "BatchNormalization":
                BN_zero_params.append(node.input[2])
                BN_zero_params.append(node.input[3])
                BN_one_params.append(node.input[1])
                BN_one_params.append(node.input[4])
        for idx,tensor in enumerate(self.onnx_model.graph.initializer):
            if tensor.name in BN_zero_params:
                if len(tensor.dims) == 1 and len(tensor.float_data) != 0:
                    for l in range(len(tensor.float_data)):
                        self.onnx_model.graph.initializer[idx].float_data[l] = 0.0
                else:
                    raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
                    raw_data_float32 = np.zeros(raw_data_float32.shape, dtype=np.float32)
                    self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()
            if tensor.name in BN_one_params:
                if len(tensor.dims) == 1 and len(tensor.float_data) != 0:
                    for l in range(len(tensor.float_data)):
                        self.onnx_model.graph.initializer[idx].float_data[l] = 1.0
                else:
                    raw_data_float32 = np.frombuffer(tensor.raw_data, dtype=np.float32)
                    raw_data_float32 = np.ones(raw_data_float32.shape, dtype=np.float32)
                    self.onnx_model.graph.initializer[idx].raw_data = raw_data_float32.tobytes()


    def load_input(self,images):
        print("==================== Load Input ====================")
        '''
        img_path = "./input/cat.png"
        self.img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        self.img = cv2.resize(self.img, (32, 32))  # (224, 224, 3)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # from BGR to RGB
        self.img = np.transpose(self.img, (2, 0, 1))  # (1, 3, 224, 224)
        self.img = np.expand_dims(self.img, 0)
        self.img = self.img.astype(np.float32)
        self.img /= 255
        '''
        self.img = images.numpy()
        print("DEBUG>>>load_input:img.dtype:",self.img.dtype)
        new_shape = (self.batch_size, -1)
        self.nn_input = np.transpose(self.img,(0, 2, 3, 1)).reshape(new_shape)
        print("DEBUG>>>load_input:nn_input.shape:",self.nn_input.shape)

    def get_ground_truth(self):
        print("==================== Get GroundTruth ====================")
        # make every node be output node
        for node in self.onnx_model.graph.node:
            for output in node.output:
                self.onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        ort_session = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())
        ort_inputs = {ort_session.get_inputs()[0].name: (self.img[0][np.newaxis,...])}#是一个字典，它将模型的输入名称映射到实际的输入数据
        # print("DEBUG:ORT_INPUTS shape = " ,ort_inputs['data'].shape)
        ort_outputs = ort_session.run(None, ort_inputs)
        # get all output node name
        node_name = [x.name for x in ort_session.get_outputs()]
        self.onnx_runtime_outs = dict(zip(node_name, ort_outputs))

        self.intermediate_result = {}
        for k,v in self.onnx_runtime_outs.items():
            if len(v.shape) == 4:
                self.intermediate_result[k] = v.transpose(0,2,3,1).flatten()
            elif len(v.shape) == 2:
                self.intermediate_result[k] = v.transpose().flatten()

class Memory:
    def __init__(self, core_num,batch_size):
        self.core_num = core_num
        self.local_memory_max_size = 262144*4 # this size is element_num. 512kB * 4 / 16bit = 262144 * 4
        self.local_memory = np.zeros((batch_size,self.core_num, self.local_memory_max_size))
        print("DEBUG:local_memory.shape = ",self.local_memory.shape)
        self.global_memory_max_size = 536870912//2 # this size is element_num. 1GB * 4/ 16bit = 536870912 * 4
        #self.global_memory_max_size = 536870912*4
        self.global_memory = np.zeros((batch_size,self.global_memory_max_size))


class Verification(ModelInfo):
    def __init__(self, model_path, pipeline_type,batch_size):
        ModelInfo.__init__(self, model_path, pipeline_type)
        self.pipeline_type = pipeline_type
        self.time_vent = 0
        self.batch_size = batch_size


    def load_compilation(self):
        print("==================== Load Compilation ====================")
        with open("../output/VerificationInfo.json", "r", encoding="utf-8") as f:
            self.FinalInfo = json.load(f)

        self.comm_pair_total_num = self.FinalInfo["comm_pair_total_num"]
        self.AG_height_start = []
        self.AG_height_end = []
        self.AG_num = len(self.FinalInfo["AG_info"])
        for AG in self.FinalInfo["AG_info"]:
            node_name = AG["node_name"]
            # print("DEUG:NODE_NAME=",node_name)
            weight_name = self.OutputToWeightDict[node_name]
            height_start = AG["height_start"]
            height_end = AG["height_end"]
            self.AG_height_start.append(height_start)
            self.AG_height_end.append(height_end)

        self.max_output_element_num = 0
        # It should be noted that the node index in FinalInfo is not exactly the same as the serial number of the ONNX model
        # This is due to the addition of input nodes and preprocessing during compilation.
        # But the node_name of both is the same.
        # Actually the node_index here is not that important as long as it is self-consistent.
        # When it comes to getting data from weight or bias, you need to pass node_name, which is unique.
        # 这里需要注意的是FinalInfo中的节点序号和ONNX模型的序号不完全一致
        # 这是由于编译时会添加输入节点以及进行预处理。但是两者的node_name是一致的。
        # 其实这里的node_index不是那么重要。只要前后自洽即可。涉及到从weight或bias拿数据时需要通过node_name，这是唯一确定的。
        self.node_name_2_index = {}
        self.node_name_2_output_element_num = {}
        for node in self.FinalInfo["node_list"]:
            output_element_num = 1
            output_dim_num = node["output_dim_num"]
            for i in range(output_dim_num):
                output_element_num = output_element_num * node["output_dim"][i]
            if self.max_output_element_num < output_element_num:
                self.max_output_element_num = output_element_num
            self.node_name_2_output_element_num[node["name"]] = output_element_num
            self.node_name_2_index[node["name"]] = node["new_node_index"]

        # For Group Conv
        for k,v in self.GEMMWeightDict.items():
            if k in self.group_weight_2_conv_name:
                OriWeight = self.GEMMWeightDict[k]
                conv_node_name = self.group_weight_2_conv_name[k]
                conv_node_index = self.node_name_2_index[conv_node_name]
                group = self.FinalInfo["node_list"][conv_node_index]["param"]["group"]
                kernel_h = self.FinalInfo["node_list"][conv_node_index]["param"]["kernel_h"]
                kernel_w = self.FinalInfo["node_list"][conv_node_index]["param"]["kernel_w"]
                full_input_channel = self.FinalInfo["node_list"][conv_node_index]["param"]["input_channel"]
                output_channel = self.FinalInfo["node_list"][conv_node_index]["param"]["output_channel"]
                group_input_channel = full_input_channel // group
                
                New_GEMM_Weight = np.zeros((kernel_h*kernel_w*full_input_channel, output_channel))
                for i in range(output_channel):
                    same_group_kernel_num = output_channel//group
                    Zero_Weight = np.zeros((kernel_h,kernel_w,full_input_channel))
                    Zero_Weight[:,:,(i//same_group_kernel_num)*group_input_channel:(i//same_group_kernel_num+1)*group_input_channel] = OriWeight[:,:,:,i]
                    Zero_Weight = Zero_Weight.flatten()
                    New_GEMM_Weight[:,i] = Zero_Weight
                self.GEMMWeightDict[k] = New_GEMM_Weight

        # Reshape Weight for FC node
        if self.pipeline_type == "element":
            if self.FinalInfo["reshape_info"] != None and "name" in self.FinalInfo["reshape_info"].keys():
                reshape_node_name = self.FinalInfo["reshape_info"]["name"]
                input_dim = self.FinalInfo["reshape_info"]["input_dim"]
                if reshape_node_name in self.OutputToWeightDict:
                    weight_name = self.OutputToWeightDict[reshape_node_name]
                    weight_matrix = self.GEMMWeightDict[weight_name]
                    new_shape = (input_dim[1], input_dim[2], input_dim[3], weight_matrix.shape[1])
                    # weight_matrix = weight_matrix.reshape((256, 6, 6, 4096))
                    weight_matrix = weight_matrix.reshape(new_shape)
                    weight_matrix = weight_matrix.transpose(1, 2, 0, 3)
                    weight_shape = weight_matrix.shape
                    self.GEMMWeightDict[weight_name] = weight_matrix.reshape(weight_matrix[:, :, :, 0].size, weight_shape[3])

        #对权重字典进行处理，方便误差计算
        self.phy_WeightDict_pos = {}
        self.phy_WeightDict_neg = {}
        self.phy_quantify = {}#存储量化的S
        self.noise_p={}
        self.noise_n={}
        for k,v in self.GEMMWeightDict.items():
            if len(v.shape)==2:
                #量化，分正负，分bit，得到新的权重词典，再simulation函数里比较两边的结果
                weight = torch.from_numpy(v).to('cuda')
                weight_range = torch.max(torch.abs(weight))
                self.phy_quantify[k] = weight_range/(2**(self.Am_precision-1)-1)
                #input量化8或16位定点
                if self.Am_precision <= 8:
                    weight_qtz = torch.round(weight / self.phy_quantify[k]).type(torch.int8)
                    A = 8
                elif self.Am_precision <=16:
                    weight_qtz = torch.round(weight / self.phy_quantify[k]).type(torch.int16)
                    A = 16
                #分正负
                weight_p = torch.nn.functional.relu(weight_qtz)
                weight_n = torch.abs(torch.subtract(weight_qtz,weight_p)).type(torch.int16)
                weight_p = weight_p.type(torch.int16)    
                #拆分bit
                weight_ps = self.split_weight(weight_p,A)
                weight_ns = self.split_weight(weight_n,A)
                
                #print("层",k,"的分离权重形状：",weight_ps.shape)
                #误差在这之后加
                self.noise_p[k] = 0
                self.noise_n[k] = 0
                if(self.sigma != 0):
                    self.noise_p[k] = torch.randn(weight_ps.shape,device = 'cuda')*self.sigma
                    self.noise_n[k] = torch.randn(weight_ps.shape,device = 'cuda')*self.sigma
                '''
                可以在这一块加入各种权重误差，漂移,,SAF等内容
                '''
                #print("未加入开关比例的weight：",weight_ps)
                #print("加入开关比例的weight：",torch.where(weight_ps>0,weight_ps,1/self.R_ratio))
                self.phy_WeightDict_pos[k] = torch.where(weight_ps>0,weight_ps,1/self.R_ratio) * self.conductance_state*(self.noise_p[k]+1)#权重的物理映射，低阻态映射，引入阻值随机噪声
                self.phy_WeightDict_neg[k] = torch.where(weight_ns>0,weight_ns,1/self.R_ratio) * self.conductance_state*(self.noise_n[k]+1)#权重的物理映射，低阻态映射
                #往后可以改一下，每一级的电导都加进去
    

        '''
        self.core_num = len(self.FinalInfo["instruction"]["core_list"])
        self.visited_single = np.zeros((1000000), dtype=np.int16)
        self.comm_index_2_index_in_core = {}
        self.comm_index_2_core_index = {}
        self.inst_num_traversal = np.zeros((self.core_num), dtype=np.int16)
        self.CoreMemory = Memory(self.core_num)
        for core_idx in range(self.core_num):
            if self.FinalInfo["instruction"]["core_list"][core_idx] != None:
                for inst_idx, instruction in enumerate(self.FinalInfo["instruction"]["core_list"][core_idx]):
                    if (instruction["operation"] == "SEND" or instruction["operation"] == "RECV"):
                        self.FinalInfo["instruction"]["core_list"][core_idx][inst_idx]["instruction_index_in_core"] = inst_idx
        '''


    def start_simulation(self,core_index, index_in_core):
        if core_index >= self.core_num:
            return
        if self.FinalInfo["instruction"]["core_list"][core_index] == None:
            next_core_index = core_index + 1
            while (not self.visited_single[next_core_index] == 0):
                next_core_index = next_core_index + 1
            self.start_simulation(next_core_index, 0)
            return

        instruction_num = len(self.FinalInfo["instruction"]["core_list"][core_index])
        self.visited_single[core_index] = 1
        for k in range(index_in_core, instruction_num):
            instruction = self.FinalInfo["instruction"]["core_list"][core_index][k]
            if instruction["operation"] == "SEND" or instruction["operation"] == "RECV":
                comm_index = instruction["comm_index"]
                instruction_index_in_core = instruction["instruction_index_in_core"]
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                if not comm_index in self.comm_index_2_index_in_core:
                    if len(self.comm_index_2_core_index) % (self.comm_pair_total_num // 20) == 0 or len(self.comm_index_2_core_index) == self.comm_pair_total_num - 1:
                        print("{:.2f}%".format(len(self.comm_index_2_core_index) / self.comm_pair_total_num * 100))
                    self.comm_index_2_index_in_core[comm_index] = instruction_index_in_core
                    self.comm_index_2_core_index[comm_index] = core_index
                    next_core_index = core_index + 1
                    while (not self.visited_single[next_core_index] == 0):
                        next_core_index = next_core_index + 1
                    self.start_simulation(next_core_index, 0)
                else:
                    corresponding_core_index = self.comm_index_2_core_index[comm_index]
                    corresponding_instruction_index_in_core = self.comm_index_2_index_in_core[comm_index]
                    element_num = instruction["element_num"]
                    if (instruction["operation"] == "RECV"):
                        destination_address = instruction["destination_address"]
                        source_address = self.FinalInfo["instruction"]["core_list"][corresponding_core_index][corresponding_instruction_index_in_core]["source_address"]
                        self.CoreMemory.local_memory[:,core_index, destination_address:destination_address + element_num] = \
                            self.CoreMemory.local_memory[:,corresponding_core_index, source_address:source_address + element_num]
                    else:
                        source_address = instruction["source_address"]
                        destination_address = self.FinalInfo["instruction"]["core_list"][corresponding_core_index][corresponding_instruction_index_in_core]["destination_address"]
                        self.CoreMemory.local_memory[:,corresponding_core_index, destination_address:destination_address + element_num] = \
                            self.CoreMemory.local_memory[:,core_index, source_address:source_address + element_num]
                    self.start_simulation(core_index, instruction_index_in_core + 1)
                    self.start_simulation(corresponding_core_index, corresponding_instruction_index_in_core + 1)
                return
            elif instruction["operation"] == "LD":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                node_name = self.FinalInfo["node_list"][node_index]["name"]
                stage = instruction["stage"]
                if self.pipeline_type == "batch":
                    if stage == "INPUT" or stage == "POST":
                        if node_index == 1:
                            self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                                self.nn_input[source_offset:source_offset + element_num]
                        else:
                            provider_index = -1 * instruction["source_address"]
                            provider = self.FinalInfo["node_list"][provider_index]["name"]
                            input_data = self.intermediate_result[provider][source_offset:source_offset + element_num]
                            if self.FinalInfo["node_list"][provider_index]["operation"] == "OP_CONV" or self.FinalInfo["node_list"][provider_index]["operation"] == "OP_FC":
                                if self.FinalInfo["node_list"][provider_index]["with_act"] == 1:
                                    input_data = (input_data + np.abs(input_data) ) / 2
                            self.CoreMemory.local_memory[core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = input_data
                elif self.pipeline_type == "element":#就先只改element
                    if stage == "INPUT":
                        # print("DEBUG: source_offset = " , source_offset)
                        # print("DEBUG: element_num = ",element_num)
                        # cut = self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num]
                        # cut2 = self.nn_input[:,source_offset:source_offset + element_num]
                        # print("DEBUG_LOAD:local memory:", cut)
                        # print("DEBUG_LOAD:nn_input:", cut2)
                        self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                            self.nn_input[:,source_offset:source_offset + element_num]
                        # print("DEBUG_LOAD:local memory:", self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num])
                if stage == "BIAS":
                    bias_name = self.OutputToBiasDict[node_name]
                    self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                        self.OriginalWeightDict[bias_name][0:element_num]
            elif instruction["operation"] == "ST":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                node_index = instruction["node_index"]
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = self.max_output_element_num * node_index
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.global_memory[
                :,destination_address + destination_offset: destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[:,core_index, source_address + source_offset: source_address + source_offset + element_num]
            elif instruction["operation"] == "MVMUL":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                AG_index = instruction["source"]
                node_name = self.FinalInfo["AG_info"][AG_index]["node_name"]

                weight_name = self.OutputToWeightDict[node_name]
                height_start = self.AG_height_start[AG_index]
                height_end = self.AG_height_end[AG_index]
                weight_matrix = self.GEMMWeightDict[weight_name][height_start:height_end + 1, :]#这里标示了权重的选取范围，如果分了比特可能要加一个维度

                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                input_element_num = instruction["input_element_num"]
                # OP1 = self.CoreMemory.local_memory[:]
                # print("DEBUG_MVMUL: input_element_num = ",input_element_num)
                # print("DEBUG_MVMUL: op1.shape = ",OP1.shape)
                # OP2 = OP1[core_index]
                # print("DEBUG_MVMUL: op2.shape = ",OP2.shape)
                # OP3 = OP2[source_address+source_offset:source_address+source_offset+input_element_num]
                # print("DEBUG_MVMUL: op3.shape = ",OP3.shape)

                input_vector = self.CoreMemory.local_memory[:,core_index,source_address+source_offset:source_address+source_offset+input_element_num]
                # print("DEBUG_MVMUL: input_vector.shape = ",input_vector.shape)
                phy_weight_p = self.phy_WeightDict_pos[weight_name][:,height_start:height_end + 1, :]
                phy_weight_n = self.phy_WeightDict_neg[weight_name][:,height_start:height_end + 1, :]
                S =self.phy_quantify[weight_name]
                #noise = [self.noise_p[weight_name][:,height_start:height_end + 1, :].to('cuda'),self.noise_n[weight_name][:,height_start:height_end + 1, :].to('cuda')]
                result_base = np.matmul(input_vector, weight_matrix)
                #physic_result = self.physical_mvm(input_vector,weight_matrix)#在这里比较两个结果，总之先写完再说
                IR_weight_p = self.IRdrop_process(phy_weight_p)
                IR_weight_n = self.IRdrop_process(phy_weight_n)
                # start = time.time()
                physic_result = self.prepared_physicalmm(input_vector,phy_weight_p,phy_weight_n,S)
                # end = time.time()
                # self.time_vent += end-start
                ir_result = self.prepared_physicalmm(input_vector, IR_weight_p,IR_weight_n,S)
              
                '''
                print("物理结果：",physic_result)
                print("逻辑结果：",result_base)
                print("误差：",np.linalg.norm(result_base - physic_result)/np.linalg.norm(result_base))
                print('\n')
                '''

                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                output_element_num = instruction["output_element_num"]
                self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + output_element_num] = result_base
            elif instruction["operation"] == "LLDI":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                imm_val = instruction["imm_value"]
                self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset: destination_address + destination_offset + element_num] = imm_val
            elif instruction["operation"] == "VVADD":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.local_memory[:, core_index,
                destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[:, core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num] + \
                    self.CoreMemory.local_memory[:, core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
            elif instruction["operation"] == "VVMUL":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                source_1 = self.CoreMemory.local_memory[:,core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num]
                source_2 = self.CoreMemory.local_memory[:,core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
                self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = source_1 * source_2
            elif instruction["operation"] == "LMV":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    self.CoreMemory.local_memory[:,core_index, source_address + source_offset:source_address + source_offset + element_num]
            elif instruction["operation"] == "VVMAX":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_1_address = instruction["source_1_address"]
                source_1_offset = instruction["source_1_offset"]
                source_2_address = instruction["source_2_address"]
                source_2_offset = instruction["source_2_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                source_1 = self.CoreMemory.local_memory[:,core_index, source_1_address + source_1_offset:source_1_address + source_1_offset + element_num]
                source_2 = self.CoreMemory.local_memory[:,core_index, source_2_address + source_2_offset:source_2_address + source_2_offset + element_num]
                self.CoreMemory.local_memory[:,core_index, destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    np.where(source_1 > source_2, source_1, source_2)
            elif instruction["operation"] == "VRELU":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                destination_address = instruction["destination_address"]
                destination_offset = instruction["destination_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                self.CoreMemory.local_memory[:,core_index,
                destination_address + destination_offset:destination_address + destination_offset + element_num] = \
                    (np.abs(self.CoreMemory.local_memory[:,core_index, source_address + source_offset:source_address + source_offset + element_num]) +
                     self.CoreMemory.local_memory[:,core_index,source_address + source_offset:source_address + source_offset + element_num]) / 2          
            elif instruction["operation"] == "VER":
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
                source_address = instruction["source_address"]
                source_offset = instruction["source_offset"]
                element_num = instruction["element_num"]
                node_index = instruction["node_index"]
                store_address = self.max_output_element_num * node_index
                input_cycle_index = instruction["input_cycle_index"]
                output_channel_element_num = self.FinalInfo["node_list"][node_index]["output_dim"][1]
                store_offset = input_cycle_index * output_channel_element_num
                self.CoreMemory.global_memory[:,store_address + store_offset:store_address + store_offset + element_num] = \
                    self.CoreMemory.local_memory[:,core_index, source_address + source_offset:source_address + source_offset + element_num]
            else:
                self.inst_num_traversal[core_index] = self.inst_num_traversal[core_index] + 1
        next_core_index = core_index + 1
        while (not self.visited_single[next_core_index] == 0):
            next_core_index = next_core_index + 1
        self.start_simulation(next_core_index, 0)

    def simulation(self):
        print("==================== Start Simulation ====================")
        #把load compilation里的那一段放到这了
        self.core_num = len(self.FinalInfo["instruction"]["core_list"])
        self.visited_single = np.zeros((1000000), dtype=np.int16) 
        self.comm_index_2_index_in_core = {}
        self.comm_index_2_core_index = {}
        self.inst_num_traversal = np.zeros((self.core_num), dtype=np.int16)
        self.CoreMemory = Memory(self.core_num,self.batch_size)
        for core_idx in range(self.core_num):
            if self.FinalInfo["instruction"]["core_list"][core_idx] != None:
                for inst_idx, instruction in enumerate(self.FinalInfo["instruction"]["core_list"][core_idx]):
                    if (instruction["operation"] == "SEND" or instruction["operation"] == "RECV"):
                        self.FinalInfo["instruction"]["core_list"][core_idx][inst_idx]["instruction_index_in_core"] = inst_idx
        #开始仿真
        self.IRdrop_counter = 0
        self.simu_counter = 0 
        self.start_simulation(0,0)
        #在simulation里加入一个返回结果
        verification_node_set = {"OP_CONV", "OP_FC"}
        for k in self.onnx_runtime_outs.keys():
            verify_node_name = k
            if verify_node_name not in self.node_name_2_index:
                continue
            verify_node_index = self.node_name_2_index[verify_node_name]
            verify_node_operation = self.FinalInfo["node_list"][verify_node_index]["operation"]
            if not verify_node_operation in verification_node_set:
                continue
            if self.FinalInfo["node_list"][verify_node_index]["output_dim_num"] == 2:
                start_element_address = verify_node_index * self.max_output_element_num
                output_element_num = self.node_name_2_output_element_num[verify_node_name]
                verify_result = self.CoreMemory.global_memory[:,start_element_address:start_element_address + output_element_num]
                ground_truth = (self.onnx_runtime_outs[verify_node_name].transpose(0, 1)).flatten()[0:output_element_num]
                result = torch.tensor(verify_result)
                truth = torch.tensor(ground_truth)
                if(verify_result[0].size == 10):
                    # print("DEBUG:result = ",result)
                    soft_out = torch.nn.functional.softmax(result)
                    # print("DEBUG:after softmax =soft_out)
                    return torch.nn.functional.softmax(result)

    def comparing(self):
        print("==================== Show Comparison  ====================")
        # verification_node_set = {"OP_CONV", "OP_FC", "OP_POOL", "OP_ELTWISE", "OP_CONCAT"}
        verification_node_set = {"OP_CONV", "OP_FC"}
        for k in self.onnx_runtime_outs.keys():
            verify_node_name = k
            if verify_node_name not in self.node_name_2_index:
                continue
            verify_node_index = self.node_name_2_index[verify_node_name]
            verify_node_operation = self.FinalInfo["node_list"][verify_node_index]["operation"]
            if not verify_node_operation in verification_node_set:
                continue
            if self.FinalInfo["node_list"][verify_node_index]["output_dim_num"] == 4:
                print(verify_node_name)
                start_element_address = verify_node_index * self.max_output_element_num
                output_channel_num = self.FinalInfo["node_list"][verify_node_index]["output_dim"][2] * \
                                     self.FinalInfo["node_list"][verify_node_index]["output_dim"][3]
                output_channel_element_num = self.FinalInfo["node_list"][verify_node_index]["output_dim"][1]
                print(output_channel_num, "*", output_channel_element_num)
                s = 0                     # start_output_channel_index
                e = output_channel_num    # end_output_channel_index
                verify_result = self.CoreMemory.global_memory[start_element_address + output_channel_element_num * s:start_element_address + output_channel_element_num * e]
                ground_truth = (self.onnx_runtime_outs[verify_node_name].transpose(0, 2, 3, 1)).flatten()[output_channel_element_num * s:output_channel_element_num * e]
                if self.FinalInfo["node_list"][verify_node_index]["operation"] == "OP_CONV":
                    if self.FinalInfo["node_list"][verify_node_index]["with_act"] == 1:
                        ground_truth = (abs(ground_truth) + ground_truth) / 2
                
                #print(verify_result-ground_truth)
                verify_result = verify_result + 1e-2
                ground_truth = ground_truth + 1e-2
                print(np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) * 100, "%")
                if np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) > 1:
                    # for channel_x in range(e):
                    #     verify_result_channel = verify_result[output_channel_element_num*channel_x:output_channel_element_num*(channel_x+1)]
                    #     ground_truth_channel = ground_truth[output_channel_element_num*channel_x:output_channel_element_num*(channel_x+1)]
                    #     if np.mean(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))) > 1:
                    #         print(channel_x)
                    #         print(np.max(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(np.argmax(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(np.mean(np.abs((verify_result_channel - ground_truth_channel) / (ground_truth_channel))))
                    #         print(verify_result_channel)
                    #         print(ground_truth_channel)
                    #         break
                    print(verify_result)
                    print(ground_truth)
                print("\n")
            elif self.FinalInfo["node_list"][verify_node_index]["output_dim_num"] == 2:
                print(verify_node_name)
                start_element_address = verify_node_index * self.max_output_element_num
                output_element_num = self.node_name_2_output_element_num[verify_node_name]
                print(output_element_num)
                verify_result = self.CoreMemory.global_memory[start_element_address:start_element_address + output_element_num]
                ground_truth = (self.onnx_runtime_outs[verify_node_name].transpose(0, 1)).flatten()[0:output_element_num]
                if self.FinalInfo["node_list"][verify_node_index]["operation"] == "OP_FC":
                    if self.FinalInfo["node_list"][verify_node_index]["with_act"] == 1:
                        ground_truth = (np.abs(ground_truth) + ground_truth) / 2
                verify_result = verify_result + 1e-2
                ground_truth = ground_truth + 1e-2
                print(np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) * 100, "%")
                if np.mean(np.abs((verify_result - ground_truth) / (ground_truth))) > 1:
                    print(verify_result)
                    print(ground_truth)
                print("\n")  

    def load_hardconfig(self):
        print("==================== Load Hardware Configuration ====================")
        with open("../config.json", "r", encoding="utf-8") as f:
            self.Hardware = json.load(f)

        self.Am_precision = self.Hardware["chip_config"]["core_config"]["matrix_config"]["adc_count"]#定点计算的精度
        print("adc_count", self.Am_precision)
        self.cell_precision = self.Hardware["chip_config"]["core_config"]["matrix_config"]["cell_precision"]#一个cell有几个bit
        self.conductance_state = self.Hardware["chip_config"]["core_config"]["matrix_config"]["reference_conductance_state"] #每个cell的阻值取值（其实是电导），有几个值取决于上面的bit数
        self.R_ratio = self.Hardware["chip_config"]["core_config"]["matrix_config"]["R_ratio"]#开关比例
        self.g_w = self.Hardware["chip_config"]["core_config"]["matrix_config"]["wordline_resistance"]#一横行的连线电阻
        self.g_b = self.Hardware["chip_config"]["core_config"]["matrix_config"]["bitline_resistance"]#一竖列的连线电阻
        self.sigma = self.Hardware["chip_config"]["core_config"]["matrix_config"]["variation"]#阻值波动的σ
        self.dac_resol = self.Hardware["chip_config"]["core_config"]["matrix_config"]["dac_resolution"]
        self.adc_resol = self.Hardware["chip_config"]["core_config"]["matrix_config"]["adc_resolution"]
        self.reference_voltage = self.Hardware["chip_config"]["core_config"]["matrix_config"]["ref_voltage"]
        #先加载这么多，有需要了再往里加

    def prepared_physicalmm(self,input,weight_p,weight_n,S):#和pysical_mvm功能完全一样，但是这里的weight是提前准备好的
        # start = time.time()
        input_vector = torch.from_numpy(input).to('cuda')
        range_1 = torch.max(torch.abs(input_vector))
        S1 = range_1/(2**(self.Am_precision-1)-1)
        S2 = S
        if self.Am_precision <= 8:
            input_qtz = torch.round(input_vector / S1).type(torch.int8)
            A = 8
            
        elif self.Am_precision <=16:
            input_qtz = torch.round(input_vector / S1).type(torch.int16)
            A = 16
        '''
        input_p = torch.nn.functional.relu(input_qtz)
        input_n = torch.abs(torch.subtract(input_qtz,input_p)).type(torch.int16)
        input_p = input_p.type(torch.int16)
        input_ps = self.split_input_com(input_p,A)
        input_ns = self.split_input_com(input_n,A)
        '''
        input_splited = self.split_input(input_qtz,A)

        # end = time.time()
        # self.time_vent+=end-start

        '''
        result0 = self.physical_xbar(input_splited[0],weight_p)
        result1 = self.physical_xbar(input_splited[0],weight_n)
        result2 = self.physical_xbar(input_splited[1],weight_p)
        result3 = self.physical_xbar(input_splited[1],weight_n)
        #执行分组计算
        sum1 = result0+result3-result1-result2
        '''
        #转化回浮点数
        sum2 = self.physical_xbar_4in1(input_splited,[weight_p,weight_n])

        #print(sum1-sum2)
        result = sum2 * S1*S2
        '''
        print("没有分正负计算:",result0)
        print("分bit计算：",result)
        print("误差：",torch.norm(result0-result)/torch.norm(result0),'\n')
        '''
        return result.cpu().numpy()
    '''
    def physical_mvm(self,vector,matrix):#之后考虑只在该函数内进行向量的量化拆分，因为权重的在外面已经做过了，这样还能解决权重已经是tensor的问题
        #ndarray 转 tensor
        input_vector = torch.from_numpy(vector).to('cuda')
        weight_matrix = torch.from_numpy(matrix).to('cuda')

        range_1 = torch.max(torch.abs(input_vector))
        range_2 = torch.max(torch.abs(weight_matrix))
        S1 = range_1/(2**(self.Am_precision-1)-1) #量化系数
        S2 = range_2/(2**(self.Am_precision-1)-1)
        
        #input量化8或16位定点
        if self.Am_precision <= 8:
            input_qtz = torch.round(input_vector / S1).type(torch.int8)
            weight_qtz = torch.round(weight_matrix / S2).type(torch.int8)
            A = 8
            result0 = np.matmul(input_qtz.cpu(),weight_qtz.cpu(),dtype=np.int32)
        elif self.Am_precision <=16:
            input_qtz = torch.round(input_vector / S1).type(torch.int16)
            weight_qtz = torch.round(weight_matrix / S2).type(torch.int16)
            A = 16
            result0 = np.matmul(input_qtz.cpu(),weight_qtz.cpu(),dtype=np.int64)#先用不分bit的来计算
        
        #print(result.dtype)

        #分正负,使用NumPy的条件索引功能来实现
        input_p = torch.nn.functional.relu(input_qtz)
        #print("input_p:",input_p)
        input_n = torch.abs(torch.subtract(input_qtz,input_p)).type(torch.int16)
        input_p = input_p.type(torch.int16)
        #print(input_n)
        weight_p = torch.nn.functional.relu(weight_qtz)
        weight_n = torch.abs(torch.subtract(weight_qtz,weight_p)).type(torch.int16)
        weight_p = weight_p.type(torch.int16)
        
        #将输入矩阵根据dac精度拆分
        input_ps = self.split_input(input_p,A)
        input_ns = self.split_input(input_n,A)

        weight_ps = self.split_weight(weight_p,A)
        weight_ns = self.split_weight(weight_n,A)
       
       

        #print("没有分bit的矩阵尺寸：",weight_p.shape)
        #print("分割后的尺寸：",weight_ps.shape)
        #print("\n")
        #模拟物理矩阵进行计算，得到四个部分和
        part0 = self.physical_xbar(input_ps,weight_ps)
        part1 = self.physical_xbar(input_ps,weight_ns)
        part2 = self.physical_xbar(input_ns,weight_ps)
        part3 = self.physical_xbar(input_ns,weight_ns)
        #执行分组计算
        #print("part0:",part0)

        sum = part0+part3-part1-part2
        #转化回浮点数
        
        result = sum * S1*S2
        result0 = result0.to('cuda')* S1 * S2 
       
        return result.cpu().numpy()
    
   '''
    def split_input(self,arr,A):#numpy
        neg_mask = torch.where(arr<0,1,0).bool()
        #print("arr:",arr)
        arr = torch.abs(arr)
        arr = arr.numpy(force=True)
        shape = arr.shape
        # print("arr:",arr[0])
        # print(arr.shape)
        bits = np.unpackbits(arr.view(np.uint8),axis = 1,bitorder='little')  # 将int8数组展开为二进制表示
        bits = bits.reshape(shape[0],shape[1], A)  # 将展开后的数组重新形状为(batchsize,lenth, 8)
        result = np.split(bits, A/self.dac_resol, axis=2)  # 按列拆分数组
        result = np.packbits(result,axis = -1,bitorder='little')
        result = np.squeeze(result)
        # print(result.shape)
        result = result.transpose((1,0,2))#至此是
        # print("DEBUG-result:",result[0])
        # print(result.shape)
        
        result = torch.from_numpy(result).to('cuda').float()
        #print("result:",result)
        result_neg=torch.zeros(result.shape,device='cuda')
        neg_mask = neg_mask.unsqueeze(1).expand_as(result)
        # print("DeBug-result_neg:",result.shape)
        # print(neg_mask.shape)
        # print(neg_mask[0])
        result_neg = torch.where(neg_mask,result,result_neg)
        result_pos = result-result_neg
        # print(result[0])
        # print("result_pos:",result_pos[0])
        # print("result_neg:",result_neg[0])
        return [result_pos,result_neg]
    
    def split_input_com(self,arr,A):#tensor实现
        #创建一个空的tensor

        splited = torch.zeros(int(A/self.dac_resol),arr.size()[0],device = 'cuda')
        mask = 2**self.dac_resol -1
        for i in range(int(A/self.dac_resol)):
            splited[i] = torch.bitwise_and(arr,mask)
            arr = torch.bitwise_right_shift(arr,self.dac_resol)
        return splited
    

    '''
    def split_weight(self,arr,A):
        # 将矩阵转换为 uint8或16 类型
        matrix_uint8 = np.ascontiguousarray(arr).view(np.uint8)
        # 拆分每个比特并重塑矩阵
        bits = np.unpackbits(matrix_uint8, axis=1,bitorder='little')
        #更改形状，将最后一维提前
        result = bits.reshape(arr.shape[0],arr.shape[1],A)
        middle = np.split(result,A/self.cell_precision,axis=2)
        matrix = np.packbits(middle,axis = 3,bitorder='little')
        result = np.squeeze(matrix)
        #result = np.asarray(unpacked_matrix)
        return result
    '''
    def split_weight(self,arr,A):#tensor实现
        #创建一个空的tensor
        splited = torch.zeros(int(A/self.cell_precision),arr.size()[0],arr.size()[1],device = 'cuda')
        mask = 2**self.cell_precision -1
        for i in range(int(A/self.cell_precision)):
            splited[i] = torch.bitwise_and(arr,mask)
            arr = torch.bitwise_right_shift(arr,self.cell_precision)
        return splited#返回形状[矩阵个数，权重长，权重宽]
    def IRdrop_process(self, weight):#处理IR-drop的模块
        xbar_num = weight.shape[0]
        m = weight.shape[1]
        n = weight.shape[2]
        Gm = torch.mean(weight,dim=(1,2)).view(-1,1)
        i = torch.arange(1,m+1).to('cuda')
        alpha = (self.g_b +(i-1)*i*Gm / 2) / (self.g_b + (m-1)*m*Gm/2)
        j = torch.arange(1,n+1).to('cuda')
        beta = (self.g_w + (n-j+1)*(n-j)*Gm / 2) / (self.g_w + (n-1)*n*Gm/2)#形状belike【8，64】

        # print("weight: ",weight.shape)
        # print("alpha: ",alpha)
        # print("beta: ",beta)
        # print("before: ",weight[0])

        weight_b = beta.unsqueeze(1) * weight
        weight_a = alpha.unsqueeze(2) * weight_b
        
        # print("after: ",weight_a[0])
        # mean = torch.norm(weight_a-weight,p=2)/torch.norm(weight,p=2)
        # if(self.IRdrop_counter == 0):
        #     self.IRdrop_mean = mean
        # else:
        #     self.IRdrop_mean = (self.IRdrop_mean*(self.IRdrop_counter)+mean)/(self.IRdrop_counter + 1)
        # self.IRdrop_counter += 1

        # print("IRdrop_mean No. ",self.IRdrop_counter,": ",self.IRdrop_mean)

        # input("----------------------------")
        # input("Press Enter to continue...")
        return weight_a
    
    def physical_xbar(self,input,weight):
        
        #物理矩阵计算,输入为（16，x）的input和（16，x，y）的weiht，16或8或什么的，这都要看dac和device的精度
        #是用cupy进行加速
        input_cycle = input.shape[0]
        phy_xbar_num = weight.shape[0]
        post_dac = input * self.reference_voltage#过一个dac
        #print("post_dac:",post_dac)
        #可以在这一块加入各种权重误差，漂移,固定误差等内容
    
        '''
        partial_sum_2 = torch.zeros(weight.shape[2]).to('cuda')
        for i in range(phy_xbar_num):
            partial_sum_1 = torch.zeros(weight.shape[2]).to('cuda')
            #内层循环为一个物理阵列上进行的计算
            for j in range(input_cycle):#不想动脑子了，两层循环得了，因为之前分比特是小尾端，所以很自然的这里就是从低比特到高比特
                partial_sum_1 += torch.matmul(post_dac[j],weight[i])*2**(j*self.dac_resol)#移位加，1bit的dac就是2的权重，2bit就是4的，类推
                #之后还有数模转换的内容
            partial_sum_2 += partial_sum_1*2**(i*self.cell_precision)#移位加部分
        partial_sum_2 = partial_sum_2/(self.reference_voltage*self.conductance_state)#相当于过了adc
        '''
        partial_sum_4 = torch.zeros(weight.shape[2],device = 'cuda')
        power_array = torch.exp2(torch.arange(input_cycle,device = 'cuda')*self.dac_resol)
        for i in range(phy_xbar_num):
            #print(i)
            
            partial_sum_3 = torch.mm(weight[i].transpose(0,1),post_dac.transpose(0,1))
            partial_sum_3 = torch.matmul(partial_sum_3,power_array)
            
            partial_sum_4 += partial_sum_3*2**(i*self.cell_precision)
        #print("partialsum_4:",partial_sum_4)
        partial_sum_4 = partial_sum_4/(self.reference_voltage*self.conductance_state)#相当于过了adc
    
        
        return partial_sum_4
    
    def physical_xbar_4in1(self,input,weight):

        start = time.time()
        input_cycle = input[0].shape[1]#加了batch之后就是[batchsize,bit_num,input]
        phy_xbar_num = weight[0].shape[0]#这个不变
        post_dac = [input[0] * self.reference_voltage,input[1]*self.reference_voltage]#过一个dac
        
        #print("input:",input)
        #print("post_dac:",post_dac,"\n")
        
        partial_sum_4 = torch.zeros(self.batch_size,4,weight[0].shape[2],device = 'cuda')
        power_array = torch.exp2(torch.arange(input_cycle,device = 'cuda')*self.dac_resol)
        power_array_weight = torch.exp2(torch.arange(phy_xbar_num,device = 'cuda')*self.cell_precision)
        
        '''
            将input的batch展开成到bit_num的维度，然后算完后再拆回独立的维度，再算位移
        '''
        for w_count in (0,1):
            for i_count in (0,1):
                # print("**********************************************")
                # print("DEBUG:weight_shape = ",weight[0].shape)
                
                cat_input = torch.cat([post_dac[i_count][i] for i in range(self.batch_size)],dim = 0)#究竟是那个维度，有待考证])
                # print("DEBUG-MVM:input : ",post_dac[i_count].shape)
                # print("DEBUG-MVM:cat_input : ",cat_input.shape)
                # print("DEBUG-MVM:weight : ",weight[w_count].shape)
                cat_weight = torch.cat([weight[w_count][i] for i in range(phy_xbar_num)],dim = 1)#究竟是那个维度，有待考证
                # print("DEBUG:cat_weight = ",cat_weight.shape)
                # cat_weight = cat_weight.view(-1,weight[0].shape[2])

                cat_sum = torch.mm(cat_weight.transpose(0,1),cat_input.transpose(0,1)) 
                
                # print("DEBUG:cat_sum = ",cat_sum)   
                
                #乘完之后是[256，80]
                cat_sum = cat_sum.view(-1,self.batch_size,input_cycle)
                # print("DEBUG:cat_sum(after) = ",cat_sum.shape)
                cat_sum = cat_sum.permute(1,0,2)#结束之后就是[10，256，8]，相当于10个batch分别算过了，接下来再考虑带batch的加权怎么算
                # print("DEBUG:位移加前的形状",cat_sum.shape)
                cat_sum = torch.matmul(cat_sum,power_array)
                # print("DEBUG:位移加后的形状",cat_sum.shape)
                # for i in range(10):
                #     Mirror_input = post_dac[i_count][i]
                #     Mirror_sum = torch.mm(cat_weight.transpose(0,1),Mirror_input.transpose(0,1))
                #     Mirror_sum = torch.matmul(Mirror_sum,power_array)
                #     print("DEBUG:compare two method ",i," : ",torch.norm(cat_sum[i]-Mirror_sum)/torch.norm(Mirror_sum))
                    
                # 现在进度是这样的，就是原本是一个输入做乘法再位移加，得到很多行一列的部分和输出，接下来要把不同crossbar分出来再做加权
                # 现在经过一系列并行操作后变成了十个这样的部分和输出，其他的都一样
                
                split_result = cat_sum.view(self.batch_size, phy_xbar_num,weight[0].shape[2])
                
                # print("DEBUG:split_result= ",split_result.shape)
                # sum_no_loop = torch.matmul(split_result.transpose(0,1),power_array_weight)
                partial_sum_4[:,w_count*2+i_count] = torch.matmul(split_result.transpose(1,2),power_array_weight)
                # print("DEBUG:partial_sum_4 = ",partial_sum_4.shape)
                # print("sum_no_loop = ",sum_no_loop)
                # print("DEBUG:循环的结果：")
                # for i in range(phy_xbar_num):
                #     partial_sum_3 = torch.mm(weight[w_count][i].transpose(0,1),post_dac[i_count].transpose(0,1))
                  
                #     partial_sum_3 = torch.matmul(partial_sum_3,power_array)
                #     # print("误差",partial_sum_3-split_result[i])
                #     partial_sum_4[w_count*2+i_count] += partial_sum_3*2**(i*self.cell_precision)
                # print("partial_sum_4:",partial_sum_4[w_count*2+i_count])
                # print("误差：",(partial_sum_4[w_count*2+i_count]-sum_no_loop)/torch.norm(sum_no_loop))
        partial_sum_5 = (partial_sum_4[:,0]+partial_sum_4[:,3]-partial_sum_4[:,1]-partial_sum_4[:,2])/(self.reference_voltage*self.conductance_state)
        end = time.time()
        self.time_vent += end-start
        
        return partial_sum_5



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIMCOMP Verification Module')
    parser.add_argument("-ModelPath", "--model_path", default="../models/ONNX/alexnet.onnx", help="onnx model path")
    parser.add_argument("-Pipeline", "--pipeline_type", default="element", help="[element] or [batch]")
    #parser.add_argument("-IsVariable","--isvariable",default = "Yes",help = "Yes or No")
    args = parser.parse_args()
    #加载验证集
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
    #transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    
    total_num = 100
    Veri_Batchsize = 100
    testloader = torch.utils.data.DataLoader(testset,batch_size = Veri_Batchsize,shuffle=False,num_workers=2)
    #性能分析
    pr = cProfile.Profile()
    
    pr.enable()
    # create verification
    verification = Verification(args.model_path, args.pipeline_type, Veri_Batchsize)
    # load model
    correct = 0
    total = 0
    total_time = 0
    verification.load_weight()
    verification.load_hardconfig()
    verification.load_compilation()
    #print("验证集大小:",len(testloader))
    for images,labels in testloader:
        print(images.shape)
        verification.load_input(images)
        verification.eliminate_no_consider_OP()
        verification.get_ground_truth()
        # loading
        
        
        # simulation
        start = time.time()
        sim_result = verification.simulation()
        end = time.time()
        total_time += end - start
        # comparison
        #verification.comparing()
        # print(sim_result)
        _, predicted = torch.max(sim_result,1)#每个batch取最大
        print("标签维度:",labels.size(0))#相当于batch_size
        print("预测结果：",predicted)
        print("标签：",labels)
        total += labels.size(0)
        print("模拟部分的平均时间：",total_time/total)
        print("访存部分平均时间",verification.time_vent/total)
        correct +=(predicted ==labels).sum().item()
        if(total % 10==0):
            print("**************************************************")
            print("验证进度：",total/total_num*100,"%")
        if (total >= total_num):
            break
    acc = 100*correct/total
    # print("全部IR_drop误差平均结果: ",verification.IRdrop_mean)
    print("准确率：",acc,"%")
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.print_stats()
    with open('perfout.txt', 'w+') as f:
        f.write(s.getvalue())