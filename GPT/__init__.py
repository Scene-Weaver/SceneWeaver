# from .check_assets import Check
# from .objaverse_retriever import ObjathorRetriever
# from .retrieve import ObjectRetriever

# global Retriever
# Retriever = ObjectRetriever()


# if __name__ == "__main__":
#     out = dict()
#     retriever = ObjectRetriever()
#     cat_lst = ["monitor"]
#     for cat in cat_lst:
#         objects = retriever.retrieve_object_by_cat(cat)
#         out[cat] = [obj[0] for obj in objects[:20]]
#     import json

#     with open("category_lst.json", "w") as f:
#         json.dump(out, f, indent=4)

#     from infinigen.assets.objaverse_assets.load_asset import load_pickled_3d_asset

#     obj_names = out[cat]
#     basedir = "~/workspace/Holodeck/data/2023_09_23/assets"
#     for obj_name in obj_names:
#         # indir = f"{basedir}/processed_2023_09_23_combine_scale"
#         filename = f"{basedir}/{obj_name}/{obj_name}.pkl.gz"
#         try:
#             obj = load_pickled_3d_asset(filename)
#             break
#         except:
#             continue
#     print("1")
