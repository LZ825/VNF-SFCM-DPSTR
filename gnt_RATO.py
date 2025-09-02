from data_pro.gnt_migration import gnt_Util
import random

def check_fulfill_requirements(dcs_dict, vnf_name):
    vnf_cpu_resource = 0
    vnf_mem_resource = 0
    dc_threshold_exceed_list = []
    for key in dcs_dict.keys():
        for vnf in dcs_dict[key]['vnfs']:
            if vnf == vnf_name:
                vnf_cpu_resource = dcs_dict[key]['vnfs'][vnf]['vnf_cpu_resource']
                vnf_mem_resource = dcs_dict[key]['vnfs'][vnf]['vnf_mem_resource']

    for key in dcs_dict.keys():
        if dcs_dict[key]['dc_cpu_resource'] > vnf_cpu_resource and dcs_dict[key]['dc_mem_resource'] > vnf_mem_resource:
            dc_threshold_exceed_list.append(key)
    return dc_threshold_exceed_list

def RATO_migration_cost(Util_function, choosed_dc, exceed_dc, need_migration_vnf):
    sfc_temp_dict = {}
    need_migration_vnf_cpu_resource = 0
    need_migration_vnf_mem_resource = 0
    SFCs = Util_function.return_sfcs()
    dcs_link_matrix = Util_function.return_dcs_link_matrix()
    for sfc_number in SFCs:
        sfc_temp = -1
        temp = -1
        dc_vnf_list = []
        for vnf in SFCs[sfc_number]['sfc_vnf']:
            dc_vnf_list.append(vnf[0])
            temp = temp + 1
            if vnf[1] == need_migration_vnf and vnf[0] == exceed_dc:
                sfc_temp = temp
                need_migration_vnf_cpu_resource = vnf[2] + need_migration_vnf_cpu_resource
                need_migration_vnf_mem_resource = vnf[3] + need_migration_vnf_mem_resource
        if sfc_temp != -1:
            if sfc_temp == len(dc_vnf_list) - 1:
                sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1]]
            elif sfc_temp == 0:
                sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp + 1]]
            else:
                sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1], dc_vnf_list[sfc_temp + 1]]
    path_length = 0
    if choosed_dc == exceed_dc:
        return 100000
    for sfc in sfc_temp_dict.keys():
        for dc in sfc_temp_dict[sfc]:
            path = Util_function.return_dijkstra_path(dcs_link_matrix, int(choosed_dc.split('dc')[-1]),
                                             int(dc.split('dc')[-1]), SFCs[sfc]['bandwidth'])
            path_length = len(path) + path_length
    return path_length* random.randint(1,10)


if __name__ == '__main__':
    text = gnt_Util.text()
    text.create_dc_link()
    text.create_all_sfc()
    text.create_all_sfc1()
    text.create_all_sfc2()
    text.create_all_sfc3()
    text.create_all_sfc4()
    text.create_all_sfc5()
    text.create_all_sfc6()
    dcs_list = []
    delay_time_list = []
    exceed_dc_list_length = []
    exceed_dc_list = text.check_dc_threshold_exceed(0.5, 0.5)
    # print(exceed_dc_list)
    while len(exceed_dc_list)!= 0:
        need_migration_vnf = text.chooose_migrate_VNF(exceed_dc_list[0])
        # print(need_migration_vnf)
        dc_meet_requirements_list = check_fulfill_requirements(text.return_dcs_dict(), need_migration_vnf)
        # print(dc_meet_requirements_list)
        path_cost = 1000
        dc_temp = None
        for choosed_dc in dc_meet_requirements_list:
            path_temp_cost = RATO_migration_cost(text, choosed_dc, exceed_dc_list[0], need_migration_vnf)
            if path_temp_cost < path_cost:
                path_cost = path_temp_cost
                dc_temp = choosed_dc
        # print("dc_temp",dc_temp)
        # print("return_dcs_dict",text.return_dcs_dict())
        dcs_list.append(dc_temp)
        path_len = text.Delay_time(dc_temp, exceed_dc_list[0], need_migration_vnf)
        print("路径长度：", path_len)
        delay_time_list.append(path_len)
        text.after_migration_dc_and_link(dc_temp, exceed_dc_list[0], need_migration_vnf)
        exceed_dc_list.pop(0)
        exceed_dc_list = text.check_dc_threshold_exceed(0.5, 0.5)
        if dc_temp == exceed_dc_list[0]:
            exceed_dc_list.pop(0)
        print("exceed_dc_list长度：",len(exceed_dc_list))
        exceed_dc_list_length.append(len(exceed_dc_list))
        # print("1212121212121212121212121",exceed_dc_list)

    print(len(dcs_list))
    print(exceed_dc_list_length)