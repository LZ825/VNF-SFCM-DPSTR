import numpy as np
import gnt_Util
import time

def search_all_dc_constrain_resource(sfcs, topology, vnf_name):
    '''选择所有符合条件的dc节点，返回一个列表'''
    dc_constrain_list = []
    vnf_name_cpu_resource = 0
    vnf_name_mem_resource = 0
    for vnf_sequence in sfcs.keys():
        for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
            if dc_vnf[1] == vnf_name:
                vnf_name_cpu_resource = dc_vnf[2]
                vnf_name_mem_resource = dc_vnf[3]
            else:
                continue
    for dc_name in topology.keys():
        if topology[dc_name]["dc_cpu_resource"] > vnf_name_cpu_resource and topology[dc_name]["dc_mem_resource"] > vnf_name_mem_resource:
            dc_constrain_list.append(dc_name)
        else:
            continue
    return dc_constrain_list

def distance_last_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
    '''当前vnf所在物理节点与服务功能链上上一个vnf所在物理节点的hop距离'''
    for vnf_sequence in sfcs.keys():
        dc_vnf_number = -1
        for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
            dc_vnf_number = dc_vnf_number + 1
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] == dc_vnf:
                distance_last_vnf = 0
                # print("这是服务功能链中第一个vnf")
                return distance_last_vnf
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] != dc_vnf:
                dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number - 1][0]
                dc2 = dc_name
                path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
                distance_last_vnf = len(path)
                return distance_last_vnf
            else:
                continue

def distance_next_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
    '''当前vnf所在物理节点与服务功能链上下一个vnf所在物理节点的hop距离'''
    for vnf_sequence in sfcs.keys():
        dc_vnf_number = -1
        for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
            dc_vnf_number = dc_vnf_number + 1
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] == dc_vnf:
                distance_last_vnf = 0
                # print("这是服务功能链中最后一个vnf")
                return distance_last_vnf
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] != dc_vnf:
                dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number + 1][0]
                dc2 = dc_name
                path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
                distance_last_vnf = len(path)
                return distance_last_vnf
            else:
                continue

if __name__ == '__main__':
    test2 = gnt_Util.text()
    test2.create_dc_link()
    dicts = test2.return_dcs_dict()
    resource_variance_list = []
    migration_cost = []
    start_time = time.time()

    test2.create_all_sfc()
    exceed_dcs = test2.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = test2.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        dc_constrain_resource_list = search_all_dc_constrain_resource(test2.return_sfcs(), test2.return_dcs_dict(),result_min)
        temp_dc = []
        temp_distance = 10000
        for dc in dc_constrain_resource_list:
            resource_cpu_mem = (dicts[dc]['dc_cpu_resource'] + dicts[dc]['dc_mem_resource'])/100
            distance = distance_next_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min, dc) + distance_last_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min, dc)
            if distance*resource_cpu_mem < temp_distance:
                temp_distance = distance*resource_cpu_mem
                temp_dc.append(dc)
        test2.migration(int(temp_dc[-1].split("dc")[-1]), key)
        migration_cost.append(temp_distance)
        variance_cpu_resource, variance_mem_resource = test2.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
    print("===========================================================================")



    test2.create_all_sfc1()
    exceed_dcs = test2.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = test2.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        dc_constrain_resource_list = search_all_dc_constrain_resource(test2.return_sfcs(), test2.return_dcs_dict(),result_min)
        temp_dc = []
        temp_distance = 1000
        for dc in dc_constrain_resource_list:
            resource_cpu_mem = (dicts[dc]['dc_cpu_resource'] + dicts[dc]['dc_mem_resource']) / 100
            distance = distance_next_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min,
                                               dc) + distance_last_vnf_in_dc(test2, test2.return_sfcs(),
                                                                             test2.return_dcs_link_matrix(), result_min,
                                                                             dc)
            if distance * resource_cpu_mem < temp_distance:
                temp_distance = distance * resource_cpu_mem
                temp_dc.append(dc)
        test2.migration(int(temp_dc[-1].split("dc")[-1]), key)
        migration_cost.append(temp_distance)
        variance_cpu_resource, variance_mem_resource = test2.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
    print("===========================================================================")



    test2.create_all_sfc2()
    exceed_dcs = test2.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = test2.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        dc_constrain_resource_list = search_all_dc_constrain_resource(test2.return_sfcs(), test2.return_dcs_dict(),result_min)
        temp_dc = []
        temp_distance = 1000
        for dc in dc_constrain_resource_list:
            resource_cpu_mem = (dicts[dc]['dc_cpu_resource'] + dicts[dc]['dc_mem_resource']) / 100
            distance = distance_next_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min,
                                               dc) + distance_last_vnf_in_dc(test2, test2.return_sfcs(),
                                                                             test2.return_dcs_link_matrix(), result_min,
                                                                             dc)
            if distance * resource_cpu_mem < temp_distance:
                temp_distance = distance * resource_cpu_mem
                temp_dc.append(dc)
        test2.migration(int(temp_dc[-1].split("dc")[-1]), key)
        migration_cost.append(temp_distance)
        variance_cpu_resource, variance_mem_resource = test2.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
    print("===========================================================================")



    test2.create_all_sfc3()
    exceed_dcs = test2.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = test2.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        dc_constrain_resource_list = search_all_dc_constrain_resource(test2.return_sfcs(), test2.return_dcs_dict(),
                                                                      result_min)
        temp_dc = []
        temp_distance = 1000
        for dc in dc_constrain_resource_list:
            resource_cpu_mem = (dicts[dc]['dc_cpu_resource'] + dicts[dc]['dc_mem_resource']) / 100
            distance = distance_next_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min,
                                               dc) + distance_last_vnf_in_dc(test2, test2.return_sfcs(),
                                                                             test2.return_dcs_link_matrix(), result_min,
                                                                             dc)
            if distance * resource_cpu_mem < temp_distance:
                temp_distance = distance * resource_cpu_mem
                temp_dc.append(dc)
        test2.migration(int(temp_dc[-1].split("dc")[-1]), key)
        migration_cost.append(temp_distance)
        variance_cpu_resource, variance_mem_resource = test2.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
    print("===========================================================================")



    test2.create_all_sfc4()
    exceed_dcs = test2.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = test2.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        dc_constrain_resource_list = search_all_dc_constrain_resource(test2.return_sfcs(), test2.return_dcs_dict(),
                                                                      result_min)
        temp_dc = []
        temp_distance = 1000
        for dc in dc_constrain_resource_list:
            resource_cpu_mem = (dicts[dc]['dc_cpu_resource'] + dicts[dc]['dc_mem_resource']) / 100
            distance = distance_next_vnf_in_dc(test2, test2.return_sfcs(), test2.return_dcs_link_matrix(), result_min,
                                               dc) + distance_last_vnf_in_dc(test2, test2.return_sfcs(),
                                                                             test2.return_dcs_link_matrix(), result_min,
                                                                             dc)
            if distance * resource_cpu_mem < temp_distance:
                temp_distance = distance * resource_cpu_mem
                temp_dc.append(dc)
        test2.migration(int(temp_dc[-1].split("dc")[-1]), key)
        migration_cost.append(temp_distance)
        variance_cpu_resource, variance_mem_resource = test2.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
    print("===========================================================================")

    end_time = time.time()
    print("程序运行时间：", end_time-start_time)
    print(resource_variance_list)
    print(migration_cost)