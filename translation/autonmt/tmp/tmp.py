# #
# # def check_datasets(base_path, datasets,  create_missing_datasets=False):
# #     # LEVEL 0: Dataset name
# #     for ds in datasets:  # Dataset
# #         ds_name = ds["name"]
# #
# #         # Check level 0 **********
# #         ds_level = os.path.join(ds_name)
# #         if not os.path.exists(os.path.join(base_path, ds_level)):
# #             print(f"- Checking dataset '{ds_level}' => Failed")
# #             continue
# #
# #         # LEVEL 1: Dataset sizes
# #         for ds_sizes in ds["sizes"]:  # Lengths
# #             ds_size_name, ds_size = ds_sizes
# #
# #             # Create dataset if asks
# #             ori_dataset = os.path.join(base_path, ds_name, "original")
# #             new_dataset = os.path.join(base_path, ds_name, ds_size_name)
# #             if not os.path.exists(new_dataset) and create_missing_datasets:  # Check that the new dataset does exists
# #                 if os.path.exists(ori_dataset):  # Check if the original folder exists
# #                     print(f"\t=> Creating missing size...")
# #                     #copy_tree(ori_dataset, new_dataset)
# #                 else:
# #                     raise IOError(f"Cannot create '{new_dataset}' because '{ori_dataset}' is missing...")
# #
# #             # Check level 1 **********
# #             ds_level = os.path.join(ds_name, ds_size_name)
# #             if not os.path.exists(os.path.join(base_path, ds_level)):
# #                 print(f"- Checking dataset '{ds_level}' => Failed")
# #                 continue
# #
# #             # LEVEL 2: Dataset languages
# #             for lang_pair in ds["languages"]:  # Languages
# #                 src_lang, trg_lang = lang_pair.split("-")
# #
# #                 # Check level 2 **********
# #                 ds_level = os.path.join(ds_name, ds_size_name, lang_pair)
# #                 if not os.path.exists(os.path.join(base_path, ds_level)):
# #                     print(f"- Checking dataset '{ds_level}' => Failed")
# #                     continue
# #
# #                 # Extra checks **********
# #                 # Check level 3: Split folder ******
# #                 ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "data", "splits")
# #                 split_path = os.path.join(base_path, ds_level)
# #                 if os.path.exists(split_path):
# #                     # Check that all split-files exist
# #                     for split in ["train", "val", "test"]:
# #                         for lang in [src_lang, trg_lang]:
# #
# #                             # Check if the split file exist
# #                             split_file = f"{split}.{lang}"
# #                             if not os.path.exists(os.path.join(base_path, ds_level, split_file)):
# #                                 print(f"- Checking dataset '{ds_level}/{split_file}' => Failed")
# #                                 continue
# #                 else:
# #                     print(f"- Checking dataset '{ds_level}' => Failed")
# #                     continue
# #
# #                 # Checks passed
# #                 ds_full_name = '_'.join([ds_name, ds_size_name, lang_pair])
# #                 print(f"- Checking dataset '{ds_full_name}' => Correct")
# #     return True
# #
# #
# #
# # def files2check(datasets):
# #     files = []
# #
# #     # LEVEL 0: Dataset name
# #     for ds in datasets:
# #         files.append(os.path.join(ds["name"]))
# #
# #         # LEVEL 1: Dataset sizes
# #         for ds_sizes in ds["sizes"]:
# #             ds_size_name, ds_size = ds_sizes
# #             files.append(os.path.join(ds["name"], ds_size_name))
# #
# #             # LEVEL 2: Dataset languages
# #             for lang_pair in ds["languages"]:
# #                 src_lang, trg_lang = lang_pair.split("-")
# #                 files.append(os.path.join(ds["name"], ds_size_name, lang_pair))
# #
# #                 # LEVEL 3: S...lhplit folder ******
# #                 files.append(os.path.join(ds["name"], ds_size_name, lang_pair, "data", "splits"))
# #
# #                 # LEVEL 3: Files
# #                 for fname in get_translation_files(src_lang, trg_lang):
# #                     files.append(os.path.join(ds["name"], ds_size_name, lang_pair, "data", "splits", fname))
# #     return files
# #
# #
# # def describe_path(path):
# #     files = []
# #     for folder, subs, dir_files in os.walk(path):
# #         folder = os.path.normpath(folder)
# #         dir_files = [os.path.normpath(os.path.join(folder, fname)) for fname in dir_files]
# #         files += [folder] + dir_files
# #     return files
#
#
# # Check level 2 **********
# ds_level = os.path.join(ds_name, ds_size_name, lang_pair)
# if not os.path.exists(os.path.join(base_path, ds_level)):
#     print(f"- Checking dataset '{ds_level}' => Failed")
#     continue
#
# # Check level 3: Split files ******
# level_folder = "splits"
# for trans_fname in trans_files:
#     # Check level 3.1: Translation files
#     ds_level = os.path.join(ds_name, ds_size_name, lang_pair, level_folder, trans_fname)
#     fname = os.path.join(base_path, ds_level)
#     if not os.path.exists(fname):
#         print(f"- Checking file '{ds_level}' => Failed")
#         continue
