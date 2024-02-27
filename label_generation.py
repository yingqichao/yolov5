import shutil
import sqlite3
# import sys
import json
 
# # adding Folder_2 to the system path
# sys.path.insert(0, '/Users/qying/ndas/maglev')

import os
# from maglev.data.generators.sqlite.sqlite_utils import get_dbpath_from_connection

def customized_image_and_label_generation(img_dir = './2mp_4cam_data/frames/5e7aa184-18af-11ec-83ea-00044bf65f70'):
    ######## PREPARE IMAGE DIRECTORY ##############
    files_list = {}
    img_files = os.walk(img_dir)
    for i, v in enumerate(img_files):
        folder_name = v[0]
        file_names = v[-1]
        if len(v)==3 and len(file_names)!=0:
            print(f"Dir found: {v}")
            for item in file_names:
                tmp = os.path.join(folder_name,item)
                key = f"{item[:item.rfind('.')]}_{tmp.split(os.sep)[4]}"
                files_list[key] = tmp
    ## test
    print("")
    print('------- TEST OS RESULT--------')
    print(len(files_list))
    for idx, item in enumerate(files_list.keys()):
        ## example: 0:./2mp_4cam_data/frames/5e7aa184-18af-11ec-83ea-00044bf65f70/camera_front_wide_120fov/crop=none/scale=half/encoding=jpeg100/isp=xavierisp/11710.jpeg
        print(f"{item}:{files_list[item]}")
        print("")
        if idx>=5:
            break

    ## clean data
    # print("")
    # print('------- CLEAN DATA --------')
    # from pathlib import Path
    # import glob
    # print(glob.glob(str(Path(img_dir) / "**" / "*.*"), recursive=True)[:5])

    class_list = [
                '\'automobile\'',
                '\'person\'',
                '\'rider\'',
                '\'animal\'',
                '\'bicycle\'',
                '\'heavy_truck\'',
    ]
    print(f"------ classes: {", ".join(class_list)}-------")

    #### sqlite route
    # (features) frame_id 
    # - (frames) id [filename: frame_number] sequence_id 
    # - (sequences) id [foldername: camera_name]
    #### example SQL left join
    # SELECT
    #     product_name,
    #     order_id
    # FROM
    #     production.products p
    # LEFT JOIN sales.order_items o ON o.product_id = p.product_id
    # ORDER BY
    #     order_id;
    sensors = None
    with sqlite3.connect('./2mp_4cam_data/dataset.sqlite') as conn:
        # item = get_dbpath_from_connection(conn)
        cur = conn.cursor()
        sensors = cur.execute(
            f"SELECT frame_number, label_name, camera_name, data FROM features \
                LEFT JOIN frames ON features.frame_id = frames.id \
                LEFT JOIN sequences ON frames.sequence_id = sequences.id \
                    WHERE features.label_name IN ({', '.join(class_list)}) \
                        AND features.label_data_type IN ('BOX2D') \
                        AND features.label_family IN ('SHAPE2D');")\
            .fetchall() ## return tuple rather than list in each cell, so does not support modification directly

    ## Test
    print("")
    print('------- TEST SQL RESULT--------')
    for i in range(3):
        ## example: (2509, 'automobile', 'camera_front_fisheye_200fov', '{"shape2d":{"box2d":{"vertices":[{"x":0.4535124,"y":0.42598686}, ...
        print(sensors[i])
        print("")
    
    ## post-process sql to get coco-like dict
    coco_style_dict = {}
    for idx, _ in enumerate(sensors):
        key = f"{sensors[idx][0]}_{sensors[idx][2]}"
        sensors[idx] = list(sensors[idx])
        cur_path = sensors[idx][0] = files_list[key]
        cur_label_cls = sensors[idx][1] = class_list.index('\''+sensors[idx][1]+'\'')
        ## sensors[idx][2] used for simplified 2d coordinate list, example: [{"x":0.26756197,"y":0.44407895},{"x":0.28357437,"y":0.4629934}]
        shape2d = json.loads(sensors[idx][3])
        if 'shape2d' not in shape2d: ## in case there is data error
            print(shape2d)
            print('Since the above record does not contain a shape 2d tag, skip this record...')
            del files_list[key]
            continue    
        shape2d = shape2d['shape2d']['box2d']['vertices']
        cur_2dpos = sensors[idx][2] = [shape2d[0]["x"], shape2d[0]["y"],
                           shape2d[1]["x"], shape2d[1]["y"]]
        
        if key not in coco_style_dict:
            ## initialize an empty label cell
            coco_style_dict[key] = [cur_path, [], []]
        coco_style_dict[key][1].append(cur_label_cls)
        coco_style_dict[key][2].append(cur_2dpos)
        
    ## Test
    print("")
    print('------- TEST refined SQL RESULT--------')
    for i in range(3):
        ## example: (2509, 'automobile', 'camera_front_fisheye_200fov', '{"shape2d":{"box2d":{"vertices":[{"x":0.4535124,"y":0.42598686}, ...
        print(sensors[i])
        print("")

    print("")
    print('------- TEST label dict RESULT--------')
    print(len(files_list))
    valid_count = 0
    for idx, item in enumerate(coco_style_dict.keys()):
        if len(coco_style_dict[item][1])>1:
            ## RULE: [path:str, label:list, 2d_pos:list]
            ## example: 7969_camera_front_fisheye_200fov:['./path/to/7969.jpeg', [0, 0, 0], [[0.059917357, 0.28947368, 0.3372934, 0.6455592], [0, 0, -0.0005165289, -0.0008223684], [0, 0, -0.0005165289, -0.0008223684]]]
            print(f"{item}:{coco_style_dict[item]}")
            print("")
            valid_count += 1
            if valid_count>=5:
                break
            
    return coco_style_dict

if __name__ == "__main__":
    import pickle
    pickle_path = './train_data.pickle'
    coco_like_dict = customized_image_and_label_generation()    
    with open(pickle_path,'wb') as t:
        pickle.dump(obj=coco_like_dict, file=t)
        print(f"write to {pickle_path} success!")




