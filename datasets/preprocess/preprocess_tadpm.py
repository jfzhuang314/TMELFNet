"""
做两件事：
1. 复制src_path下方的每个子目录的ori目录下的文件到dest_path下方的对应子目录中；
2. 根据train-test-split.txt文件中的train和test部分，生成train.txt和test.txt文件；
"""

import re
import shutil
from pathlib import Path


def do_copy_ori_folds():
    """
    src_path下方有多个子目录，每个子目录下有ori和final两个子目录；
    要求：将src_path下方的每个子目录的ori目录下的文件复制到dest_path下方的对应子目录中；
    新文件的命名规则：在原文件名前添加subdir.name_前缀
    """
    def copy_ori_files(src_path, dest_path):
        src_path = Path(src_path)
        dest_path = Path(dest_path)

        # 确保目标路径存在
        dest_path.mkdir(parents=True, exist_ok=True)

        # 遍历源路径下的所有子目录
        for subdir in src_path.iterdir():
            if subdir.is_dir():
                ori_path = subdir / "ori"

                # 检查ori目录是否存在
                if ori_path.exists() and ori_path.is_dir():
                    # 创建目标子目录
                    dest_subdir = dest_path / subdir.name / "ori"
                    dest_subdir.mkdir(parents=True, exist_ok=True)

                    # 复制ori目录下的所有文件
                    copied_count = 0
                    for file_path in ori_path.iterdir():
                        if file_path.is_file():
                            # 构造新的文件名：子目录名_原文件名
                            new_filename = f"{subdir.name}_{file_path.name}"
                            dest_file = dest_subdir / new_filename
                            # dest_file = dest_subdir / file_path.name
                            shutil.copy2(file_path, dest_file)
                            copied_count += 1
                            print(f"已复制: {file_path} -> {dest_file}")

                    print(f"子目录 {subdir.name} 完成，共复制 {copied_count} 个文件")
                else:
                    print(f"警告: 子目录 {subdir.name} 下没有ori目录")

    src_path = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset"
    dest_path = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori"


    # 执行复制操作
    print(f"开始从 {src_path} 复制文件到 {dest_path}")
    copy_ori_files(src_path, dest_path)
    print("文件复制完成！")

def do_build_test_txt():
    """
    从train-test-split.txt文件中提取测试集文件名，并写入testing_full.txt文件
    train-test-split.txt文件中有“train”及"test"两个部分，每个部分有多个文件名；
    每个文件名可与src_path下方的子目录名匹配；
    1. 根据train/test及下方的文件名，提取ori目录下的obj文件,并写入dest_split_train_path或dest_split_test_path；
    """

    def parse_train_test_split_json(split_content):
        """
        解析JSON格式的train-test-split内容

        Args:
            split_content (str): train-test-split.txt文件内容

        Returns:
            tuple: (train_files_list, test_files_list)
        """
        # 使用正则表达式提取train和test部分
        train_match = re.search(r'"train"\s*:\s*\{([^}]+)\}', split_content)
        test_match = re.search(r'"test"\s*:\s*\{([^}]+)\}', split_content)

        train_files = []
        test_files = []

        if train_match:
            train_numbers = train_match.group(1)
            # 提取数字并补零到4位
            train_files = [f"{int(num):04d}" for num in re.findall(r'\d+', train_numbers)]

        if test_match:
            test_numbers = test_match.group(1)
            # 提取数字并补零到4位
            test_files = [f"{int(num):04d}" for num in re.findall(r'\d+', test_numbers)]

        return train_files, test_files

    def extract_stl_files_by_split(src_path, split_file_path, dest_train_path, dest_test_path):
        """
        根据train-test-split.txt中的文件名，从对应子目录的ori目录中提取特定格式的STL文件
        只保留 {dirname}_L_Ori.stl 或 {dirname}_R_Ori.stl 格式的文件

        Args:
            src_path (str): 源数据路径
            split_file_path (str): train-test-split.txt文件路径
            dest_train_path (str): 训练集文件列表输出路径
            dest_test_path (str): 测试集文件列表输出路径
        """
        src_path = Path(src_path)

        # 读取split文件内容
        with open(split_file_path, 'r', encoding='utf-8') as f:
            split_content = f.read()

        # 解析split文件
        train_files, test_files = parse_train_test_split_json(split_content)

        print(f"找到训练集文件: {len(train_files)} 个")
        print(f"找到测试集文件: {len(test_files)} 个")

        # 收集训练集和测试集的stl文件路径
        train_stl_paths = []
        test_stl_paths = []

        # 处理训练集文件
        print("\n处理训练集文件...")
        for filename in train_files:
            ori_path = src_path / filename / "ori"
            if ori_path.exists():
                # 查找符合格式的STL文件: {dirname}_L_Ori.stl 或 {dirname}_R_Ori.stl
                l_stl = ori_path / f"{filename}_L_Ori.stl"
                r_stl = ori_path / f"{filename}_U_Ori.stl"

                found_files = []
                if l_stl.exists():
                    train_stl_paths.append(str(l_stl))
                    found_files.append(l_stl.name)
                if r_stl.exists():
                    train_stl_paths.append(str(r_stl))
                    found_files.append(r_stl.name)

                if found_files:
                    print(f"✓ 训练集: {filename} -> {', '.join(found_files)}")
                else:
                    print(f"⚠ 警告: 训练集 {filename} 目录下没有找到符合条件的STL文件")
            else:
                print(f"⚠ 警告: 训练集子目录 {filename}/ori 不存在")

        # 处理测试集文件
        print("\n处理测试集文件...")
        for filename in test_files:
            ori_path = src_path / filename / "ori"
            if ori_path.exists():
                # 查找符合格式的STL文件: {dirname}_L_Ori.stl 或 {dirname}_R_Ori.stl
                l_stl = ori_path / f"{filename}_L_Ori.stl"
                r_stl = ori_path / f"{filename}_U_Ori.stl"

                found_files = []
                if l_stl.exists():
                    test_stl_paths.append(str(l_stl))
                    found_files.append(l_stl.name)
                if r_stl.exists():
                    test_stl_paths.append(str(r_stl))
                    found_files.append(r_stl.name)

                if found_files:
                    print(f"✓ 测试集: {filename} -> {', '.join(found_files)}")
                else:
                    print(f"⚠ 警告: 测试集 {filename} 目录下没有找到符合条件的STL文件")
            else:
                print(f"⚠ 警告: 测试集子目录 {filename}/ori 不存在")

        # 写入训练集文件列表
        with open(dest_train_path, 'w', encoding='utf-8') as f:
            for stl_path in train_stl_paths:
                f.write(Path(stl_path).name + '\n')

        # 写入测试集文件列表
        with open(dest_test_path, 'w', encoding='utf-8') as f:
            for stl_path in test_stl_paths:
                f.write(Path(stl_path).name + '\n')

        print(f"\n🎉 完成!")
        print(f"训练集STL文件列表已写入: {dest_train_path}")
        print(f"  - 共 {len(train_stl_paths)} 个STL文件")
        print(f"测试集STL文件列表已写入: {dest_test_path}")
        print(f"  - 共 {len(test_stl_paths)} 个STL文件")

    src_path = r"E:\mesh_datasets\3dTadpm\Orthodontic_dental_dataset_ori"
    src_split_path = r"E:\mesh_datasets\3dTadpm\train-test-split.txt"
    dest_split_train_path = r"E:\mesh_datasets\3dTadpm\training_full.txt"
    dest_split_test_path = r"E:\mesh_datasets\3dTadpm\testing_full.txt"

    # 执行文件提取操作
    print("开始处理train-test-split.txt文件...")
    extract_stl_files_by_split(src_path, src_split_path, dest_split_train_path, dest_split_test_path)

def main():
    #########################################
    # 仅仅复制ori文件
    # do_copy_ori_folds()
    do_build_test_txt()

if __name__ == '__main__':

    main()