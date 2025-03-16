 #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XML标注文件转YOLO格式TXT工具
将XML格式的目标检测标注文件转换为YOLO格式的TXT文件

XML格式: 边界框为左上角和右下角坐标 (xmin, ymin, xmax, ymax)
YOLO格式: 归一化的中心点坐标和宽高 (x_center, y_center, width, height)
"""

import xml.etree.ElementTree as ET
import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm


class XMLtoYOLOConverter:
    """XML标注文件转YOLO格式TXT转换器"""
    
    def __init__(self, input_dir, output_dir, classes):
        """
        初始化转换器
        
        Args:
            input_dir (str): 输入XML文件目录
            output_dir (str): 输出TXT文件目录
            classes (list): 类别列表
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.classes = classes
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def convert_coordinates(self, image_size, box):
        """
        将XML中的边界框坐标转换为YOLO格式的归一化坐标
        
        Args:
            image_size (tuple): 图像尺寸 (width, height)
            box (tuple): 边界框坐标 (xmin, xmax, ymin, ymax)
            
        Returns:
            tuple: YOLO格式的坐标 (x_center, y_center, width, height)
        """
        dw = 1.0 / image_size[0]
        dh = 1.0 / image_size[1]
        
        # 计算中心点坐标
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        
        # 计算宽高
        w = box[1] - box[0]
        h = box[3] - box[2]
        
        # 归一化
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        
        return (x, y, w, h)
    
    def convert_annotation(self, xml_file):
        """
        转换单个XML文件到TXT文件
        
        Args:
            xml_file (str): XML文件路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 获取文件名（不含扩展名）
            file_name = os.path.basename(xml_file)[:-4]
            
            # 打开输入和输出文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            output_file = os.path.join(self.output_dir, f"{file_name}.txt")
            with open(output_file, 'w', encoding='UTF-8') as out_file:
                # 获取图像尺寸
                size = root.find('size')
                if size is None:
                    print(f"警告: 文件 {xml_file} 中未找到尺寸信息")
                    return False
                    
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # 处理每个目标
                for obj in root.iter('object'):
                    # 获取困难度标志，跳过困难样本
                    difficult = obj.find('difficult')
                    if difficult is not None and int(difficult.text) == 1:
                        continue
                    
                    # 获取类别
                    cls = obj.find('name').text
                    if cls not in self.classes:
                        print(f"警告: 类别 '{cls}' 不在预定义类别列表中，跳过")
                        continue
                    
                    cls_id = self.classes.index(cls)
                    
                    # 获取边界框
                    xmlbox = obj.find('bndbox')
                    if xmlbox is None:
                        continue
                        
                    # 提取边界框坐标
                    try:
                        xmin = float(xmlbox.find('xmin').text)
                        xmax = float(xmlbox.find('xmax').text)
                        ymin = float(xmlbox.find('ymin').text)
                        ymax = float(xmlbox.find('ymax').text)
                    except (AttributeError, ValueError) as e:
                        print(f"警告: 文件 {xml_file} 中的边界框数据无效: {e}")
                        continue
                    
                    # 转换坐标
                    box = (xmin, xmax, ymin, ymax)
                    bb = self.convert_coordinates((width, height), box)
                    
                    # 写入TXT文件
                    out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
                
            return True
            
        except Exception as e:
            print(f"错误: 处理文件 {xml_file} 时发生异常: {e}")
            return False
    
    def convert_all(self):
        """
        转换所有XML文件到TXT文件
        
        Returns:
            tuple: (成功计数, 总文件数)
        """
        # 获取所有XML文件
        xml_files = glob.glob(os.path.join(self.input_dir, '*.xml'))
        total_files = len(xml_files)
        
        if total_files == 0:
            print(f"警告: 在 {self.input_dir} 中没有找到XML文件")
            return 0, 0
        
        print(f"\n找到 {total_files} 个XML文件待转换")
        
        # 使用tqdm显示进度条
        success_count = 0
        for xml_file in tqdm(xml_files, desc="转换进度"):
            if self.convert_annotation(xml_file):
                success_count += 1
        
        return success_count, total_files


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将XML标注文件转换为YOLO格式的TXT文件')
    parser.add_argument('--input', default='Annotations', help='输入XML文件目录 (默认: Annotations)')
    parser.add_argument('--output', default='labels', help='输出TXT文件目录 (默认: labels)')
    parser.add_argument('--classes', default='battery,drug,plastic,metal,potato,carrot,daikon,brick,cobblestone,cup',
                       help='类别列表，用逗号分隔 (默认: battery,drug,plastic,metal,potato,carrot,daikon,brick,cobblestone,cup)')
    args = parser.parse_args()
    
    # 处理类别列表
    classes = args.classes.split(',')
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 构建输入和输出路径
    input_dir = os.path.join(current_dir, args.input)
    output_dir = os.path.join(current_dir, args.output)
    
    print(f"XML文件目录: {input_dir}")
    print(f"TXT文件输出目录: {output_dir}")
    print(f"类别列表: {classes}")
    
    # 创建转换器并执行转换
    converter = XMLtoYOLOConverter(input_dir, output_dir, classes)
    success_count, total_files = converter.convert_all()
    
    # 打印结果
    print(f"\n转换完成!")
    print(f"成功: {success_count}/{total_files} 个文件")
    if success_count < total_files:
        print(f"失败: {total_files - success_count} 个文件")


if __name__ == "__main__":
    main()