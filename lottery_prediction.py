def generate_report(self, top_n=10):
        """
        Tạo báo cáo dự đoán đầy đủ.
        
        Args:
            top_n (int): Số lượng số cần dự đoán.
            
        Returns:
            dict: Từ điển chứa kết quả dự đoán từ các phương pháp.
        """
        print("\n📋 Tạo báo cáo dự đoán...")
        
        # Lấy ngày gần nhất và dự đoán cho ngày tiếp theo
        last_date = self.analyzer.processed_data['date'].max()
        next_date = last_date + timedelta(days=1)
        
        print(f"\n🗓️ Dự báo kết quả ngày {next_date.day} tháng {next_date.month} năm {next_date.year}")
        print("=" * 50)
        
        # Dictionary để lưu kết quả từ các phương pháp
        results = {}
        
        # Tổng số bản ghi để tính phần trăm
        total_records = len(self.analyzer.processed_data)
        
        # 1. Dự đoán theo tần suất
        freq_predictions = self.predict_by_frequency(top_n=top_n)
        results['Tần suất'] = freq_predictions
        
        # Tính tổng số lần xuất hiện để tính phần trăm
        total_appearances = sum([count for _, count in freq_predictions])
        
        print("\n📊 Kết quả dự đoán theo tần suất:")
        print("-" * 40)
        print(f"{'Số':^5} | {'Tần suất':^10} | {'Phần trăm (%)':<15}")
        print("-" * 40)
        for num, count in freq_predictions:
            percentage = (count / total_appearances) * 100
            print(f"{num:^5} | {count:^10} | {percentage:>15.2f}%")
        
        # 2. Dự đoán theo mẫu
        pattern_predictions = self.predict_by_pattern(top_n=top_n)
        results['Mẫu'] = pattern_predictions
        
        # Tính tổng điểm để tính phần trăm
        total_weight = sum([weight for _, weight in pattern_predictions])
        
        print("\n🔍 Kết quả dự đoán theo mẫu:")
        print("-" * 40)
        print(f"{'Số':^5} | {'Điểm':^10} | {'Phần trăm (%)':<15}")
        print("-" * 40)
        for num, weight in pattern_predictions:
            percentage = (weight / total_weight) * 100
            print(f"{num:^5} | {weight:^10.2f} | {percentage:>15.2f}%")
        
        # 3. Dự đoán theo học máy (nếu có)
        if self.ml_model is not None:
            ml_predictions = self.predict_by_machine_learning(top_n=top_n)
            results['Học máy'] = ml_predictions
            
            print("\n🤖 Kết quả dự đoán theo học máy:")
            print("-" * 40)
            print(f"{'Số':^5} | {'Xác suất':^10} | {'Phần trăm (%)':<15}")
            print("-" * 40)
            for num, prob in ml_predictions:
                print(f"{num:^5} | {prob*100:^10.2f}% | {prob*100:>15.2f}%")
        
        # 4. Dự đoán theo ensemble (nếu có)
        if self.ensemble_model is not None:
            ensemble_predictions = self.predict_by_ensemble(top_n=top_n)
            results['Tổng hợp'] = ensemble_predictions
            
            print("\n🔄 Kết quả dự đoán theo mô hình tổng hợp:")
            print("-" * 40)
            print(f"{'Số':^5} | {'Xác suất':^10} | {'Phần trăm (%)':<15}")
            print("-" * 40)
            for num, prob in ensemble_predictions:
                print(f"{num:^5} | {prob*100:^10.2f}% | {prob*100:>15.2f}%")
        
        # 5. Dự đoán theo LSTM (nếu có)
        if PYTORCH_AVAILABLE and self.lstm_model is not None:
            lstm_predictions = self.predict_by_lstm(top_n=top_n)
            results['LSTM'] = lstm_predictions
            
            print("\n🧠 Kết quả dự đoán theo LSTM:")
            print("-" * 40)
            print(f"{'Số':^5} | {'Xác suất':^10} | {'Phần trăm (%)':<15}")
            print("-" * 40)
            for num, prob in lstm_predictions:
                print(f"{num:^5} | {prob*100:^10.2f}% | {prob*100:>15.2f}%")
        
        # 6. Kết hợp tất cả các phương pháp
        combined_predictions = self.combine_predictions(top_n=top_n)
        results['Kết hợp'] = combined_predictions
        
        # Tính tổng điểm để tính phần trăm
        total_score = sum([score for _, score in combined_predictions])
        
        print("\n🌟 KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG:")
        print("=" * 40)
        print(f"{'Số':^5} | {'Điểm':^10} | {'Phần trăm (%)':<15}")
        print("=" * 40)
        for num, score in combined_predictions:
            percentage = (score / total_score) * 100
            print(f"{num:^5} | {score:^10.4f} | {percentage:>15.2f}%")
        
        # Trực quan hóa kết quả
        self.visualize_results(results)
        
        # Lưu kết quả dự đoán ra file
        self._save_prediction_results(results, next_date)
        
        return results
    
    def _save_prediction_results(self, results, prediction_date):
        """
        Lưu kết quả dự đoán ra file.
        
        Args:
            results (dict): Từ điển chứa kết quả dự đoán.
            prediction_date (datetime): Ngày dự đoán.
        """
        # Tạo thư mục lưu kết quả nếu chưa tồn tại
        results_dir = "predictions"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Tên file kết quả
        filename = os.path.join(results_dir, f"prediction_{prediction_date.strftime('%Y-%m-%d')}.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Tiêu đề
            f.write(f"DỰ BÁO KẾT QUẢ NGÀY {prediction_date.day} THÁNG {prediction_date.month} NĂM {prediction_date.year}\n")
            f.write("=" * 50 + "\n\n")
            
            # Ghi kết quả từng phương pháp
            for method, preds in results.items():
                f.write(f"{method.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                if method in ['Học máy', 'LSTM', 'Tổng hợp']:
                    f.write(f"{'Số':^5} | {'Xác suất':^10} | {'Phần trăm (%)':<15}\n")
                    total = sum([prob for _, prob in preds])
                    for num, prob in preds:
                        percentage = (prob / total) * 100 if total > 0 else 0
                        f.write(f"{num:^5} | {prob*100:^10.2f}% | {percentage:>15.2f}%\n")
                else:
                    if method == 'Tần suất':
                        f.write(f"{'Số':^5} | {'Tần suất':^10} | {'Phần trăm (%)':<15}\n")
                        total = sum([count for _, count in preds])
                        for num, count in preds:
                            percentage = (count / total) * 100 if total > 0 else 0
                            f.write(f"{num:^5} | {count:^10} | {percentage:>15.2f}%\n")
                    else:
                        f.write(f"{'Số':^5} | {'Điểm':^10} | {'Phần trăm (%)':<15}\n")
                        total = sum([score for _, score in preds])
                        for num, score in preds:
                            percentage = (score / total) * 100 if total > 0 else 0
                            f.write(f"{num:^5} | {score:^10.4f} | {percentage:>15.2f}%\n")
                
                f.write("\n")
            
            # Ghi kết quả cuối cùng
            f.write("\nKẾT QUẢ DỰ ĐOÁN CUỐI CÙNG:\n")
            f.write("=" * 40 + "\n")
            f.write(f"{'Số':^5} | {'Điểm':^10} | {'Phần trăm (%)':<15}\n")
            f.write("=" * 40 + "\n")
            
            combined = results.get('Kết hợp', [])
            total_score = sum([score for _, score in combined])
            
            for num, score in combined:
                percentage = (score / total_score) * 100 if total_score > 0 else 0
                f.write(f"{num:^5} | {score:^10.4f} | {percentage:>15.2f}%\n")
        
        print(f"\n💾 Đã lưu kết quả dự đoán vào file {filename}")


def main():
    """
    Hàm chính để chạy chương trình.
    """
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description='Dự đoán kết quả xổ số dựa trên dữ liệu lịch sử.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', '-d', required=True,
                        help='Đường dẫn đến file JSON chứa dữ liệu lịch sử.')
    
    parser.add_argument('--top', '-t', type=int, default=10,
                        help='Số lượng con số muốn dự đoán.')
    
    parser.add_argument('--train', '-tr', action='store_true',
                        help='Huấn luyện các mô hình học máy.')
    
    parser.add_argument('--load', '-l', action='store_true',
                        help='Tải các mô hình đã lưu.')
    
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Hiển thị biểu đồ phân tích dữ liệu.')
    
    parser.add_argument('--hot', type=int, default=0,
                        help='Hiển thị N số nóng (số xuất hiện nhiều nhất).')
    
    parser.add_argument('--cold', type=int, default=0,
                        help='Hiển thị N số lạnh (số lâu không xuất hiện).')
    
    parser.add_argument('--pairs', type=int, default=0,
                        help='Hiển thị N cặp số hay xuất hiện cùng nhau.')
    
    parser.add_argument('--use-lstm', action='store_true',
                        help='Cố gắng sử dụng mô hình LSTM (yêu cầu PyTorch).')
    
    args = parser.parse_args()
    
    try:
        # Hiển thị thông tin tiến trình
        print("🚀 Bắt đầu chương trình dự đoán xổ số...")
        print(f"📂 Đường dẫn dữ liệu: {args.data}")
        print(f"🔢 Số lượng dự đoán: {args.top}")
        print("=" * 50)
        
        # Tạo đối tượng phân tích
        analyzer = LotteryAnalyzer(args.data)
        
        # Hiển thị % tiến trình
        print("⏳ Tiến trình: 10% - Đọc dữ liệu")
        analyzer.load_data()
        
        print("⏳ Tiến trình: 25% - Xử lý dữ liệu")
        analyzer.preprocess_data()
        
        # Hiển thị thông tin phân tích nếu cần
        if args.visualize:
            print("⏳ Tiến trình: 35% - Trực quan hóa dữ liệu")
            analyzer.visualize_number_frequency()
            
            # Phân tích phân phối các số
            distribution = analyzer.analyze_number_distribution()
            
            # Hiển thị phân phối theo chữ số đầu
            plt.figure(figsize=(10, 6))
            first_digits = sorted(distribution['first_digit'].items())
            plt.bar([x[0] for x in first_digits], [x[1] for x in first_digits], color='seagreen')
            plt.title('Phân phối theo chữ số đầu', fontsize=14)
            plt.xlabel('Chữ số đầu', fontsize=12)
            plt.ylabel('Tần suất', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
            
            # Hiển thị phân phối theo chữ số cuối
            plt.figure(figsize=(10, 6))
            second_digits = sorted(distribution['second_digit'].items())
            plt.bar([x[0] for x in second_digits], [x[1] for x in second_digits], color='indianred')
            plt.title('Phân phối theo chữ số cuối', fontsize=14)
            plt.xlabel('Chữ số cuối', fontsize=12)
            plt.ylabel('Tần suất', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
            
            # Hiển thị phân phối theo tổng chữ số
            plt.figure(figsize=(12, 6))
            sum_digits = sorted(distribution['sum_digits'].items())
            plt.bar([x[0] for x in sum_digits], [x[1] for x in sum_digits], color='royalblue')
            plt.title('Phân phối theo tổng chữ số', fontsize=14)
            plt.xlabel('Tổng chữ số', fontsize=12)
            plt.ylabel('Tần suất', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
            
            # Hiển thị phân phối chẵn/lẻ
            plt.figure(figsize=(8, 6))
            even_odd = distribution['even_odd']
            plt.pie([even_odd['even'], even_odd['odd'], even_odd['mixed']],
                    labels=['Chẵn-Chẵn', 'Lẻ-Lẻ', 'Hỗn hợp'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['lightcoral', 'lightblue', 'lightgreen'])
            plt.title('Phân phối số theo tính chất chẵn/lẻ', fontsize=14)
            plt.show()
            
            # Hiển thị xu hướng của 5 số phổ biến
            hot_numbers = [x[0] for x in analyzer.get_hot_numbers(5)]
            analyzer.visualize_trends(hot_numbers)
        
        # Hiển thị số nóng nếu cần
        if args.hot > 0:
            print("⏳ Tiến trình: 40% - Phân tích số nóng")
            hot_numbers = analyzer.get_hot_numbers(args.hot)
            print("\n🔥 Các số nóng (xuất hiện nhiều nhất):")
            print("-" * 30)
            print(f"{'Số':^5} | {'Tần suất':^10}")
            print("-" * 30)
            for num, count in hot_numbers:
                print(f"{num:^5} | {count:^10}")
        
        # Hiển thị số lạnh nếu cần
        if args.cold > 0:
            print("⏳ Tiến trình: 45% - Phân tích số lạnh")
            cold_numbers = analyzer.get_cold_numbers(args.cold)
            print("\n❄️ Các số lạnh (lâu không xuất hiện):")
            print("-" * 30)
            print(f"{'Số':^5} | {'Số ngày':^10}")
            print("-" * 30)
            for num, days in cold_numbers:
                if days == float('inf'):
                    print(f"{num:^5} | {'Chưa xuất hiện':^10}")
                else:
                    print(f"{num:^5} | {days:^10}")
        
        # Hiển thị cặp số nếu cần
        if args.pairs > 0:
            print("⏳ Tiến trình: 50% - Phân tích cặp số")
            hot_pairs = analyzer.get_hot_pairs(args.pairs)
            print("\n👫 Các cặp số hay xuất hiện cùng nhau:")
            print("-" * 30)
            print(f"{'Cặp số':^10} | {'Tần suất':^10}")
            print("-" * 30)
            for pair, count in hot_pairs:
                print(f"{pair[0]}-{pair[1]:^10} | {count:^10}")
        
        # Tạo đối tượng dự đoán
        print("⏳ Tiến trình: 55% - Khởi tạo mô hình dự đoán")
        predictor = LotteryPredictor(analyzer)
        
        # Tải mô hình nếu cần
        if args.load:
            print("⏳ Tiến trình: 60% - Tải mô hình đã lưu")
            predictor.load_models()
        
        # Huấn luyện mô hình nếu cần
        if args.train:
            print("⏳ Tiến trình: 65% - Huấn luyện mô hình học máy")
            # Huấn luyện mô hình học máy
            predictor.train_machine_learning_model()
            
            print("⏳ Tiến trình: 75% - Tạo mô hình tổng hợp")
            # Tạo mô hình tổng hợp
            predictor.create_ensemble_model()
            
            # Huấn luyện mô hình LSTM nếu có
            if args.use_lstm and PYTORCH_AVAILABLE:
                print("⏳ Tiến trình: 85% - Huấn luyện mô hình LSTM")
                predictor.train_lstm_model()
        
        # Tạo báo cáo dự đoán
        print("⏳ Tiến trình: 95% - Tạo báo cáo dự đoán")
        predictor.generate_report(args.top)
        
        print("\n✅ Tiến trình: 100% - Hoàn thành!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chương trình dự đoán kết quả xổ số nâng cao sử dụng nhiều phương pháp học máy và phân tích khác nhau.
"""

import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Thư viện học máy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Kiểm tra PyTorch nhưng không cố nhập nếu có lỗi
PYTORCH_AVAILABLE = False
try:
    # Chỉ thử nhập PyTorch nếu người dùng chỉ định sử dụng LSTM
    if '--use-lstm' in os.sys.argv:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader, TensorDataset
        PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch không khả dụng hoặc không tương thích. Chương trình sẽ không sử dụng mô hình LSTM.")
    PYTORCH_AVAILABLE = False

class LotteryAnalyzer:
    """
    Lớp phân tích dữ liệu xổ số và tạo đặc trưng.
    """
    
    def __init__(self, data_path):
        """
        Khởi tạo đối tượng LotteryAnalyzer.
        
        Args:
            data_path (str): Đường dẫn đến file JSON chứa dữ liệu lịch sử xổ số.
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.number_frequency = None
        self.combinations = None
        self.prize_columns = [
            'prize1', 
            'prize2_1', 'prize2_2',
            'prize3_1', 'prize3_2', 'prize3_3', 'prize3_4', 'prize3_5', 'prize3_6',
            'prize4_1', 'prize4_2', 'prize4_3', 'prize4_4',
            'prize5_1', 'prize5_2', 'prize5_3', 'prize5_4', 'prize5_5', 'prize5_6',
            'prize6_1', 'prize6_2', 'prize6_3',
            'prize7_1', 'prize7_2', 'prize7_3', 'prize7_4'
        ]
        
    def load_data(self):
        """
        Đọc dữ liệu từ file JSON và chuyển thành DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame chứa dữ liệu xổ số.
        """
        print(f"\n🔍 Đang đọc dữ liệu từ file {self.data_path}...")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Chuyển dữ liệu thành DataFrame
            self.raw_data = pd.DataFrame(data)
            
            # Chuyển cột date sang định dạng datetime
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
            
            # Sắp xếp dữ liệu theo ngày (từ cũ đến mới)
            self.raw_data.sort_values('date', inplace=True)
            
            print(f"✅ Đã đọc thành công {len(self.raw_data)} bản ghi.")
            print(f"📅 Dữ liệu từ {self.raw_data['date'].min().strftime('%Y-%m-%d')} đến {self.raw_data['date'].max().strftime('%Y-%m-%d')}")
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc file JSON: {e}")
            return None
    
    def preprocess_data(self):
        """
        Xử lý dữ liệu, trích xuất 2 số cuối của tất cả các giải thưởng.
        
        Returns:
            pandas.DataFrame: DataFrame chứa dữ liệu đã xử lý.
        """
        if self.raw_data is None:
            print("❌ Chưa có dữ liệu. Vui lòng gọi phương thức load_data() trước.")
            return None
        
        print("\n📊 Đang xử lý dữ liệu...")
        
        # Tạo danh sách chứa tất cả các số (2 chữ số cuối)
        all_numbers = []
        all_combinations = []
        
        # DataFrame để lưu dữ liệu đã xử lý
        processed_df = pd.DataFrame()
        processed_df['date'] = self.raw_data['date']
        
        # Trích xuất 2 số cuối của mỗi giải thưởng
        for column in self.prize_columns:
            # Lấy 2 số cuối
            processed_df[f"{column}_last2"] = self.raw_data[column].astype(str).str.zfill(2).str[-2:]
            
            # Thêm vào danh sách tất cả các số
            all_numbers.extend(processed_df[f"{column}_last2"].tolist())
        
        # Tính tần suất xuất hiện của mỗi số
        self.number_frequency = Counter(all_numbers)
        
        # Tạo danh sách tất cả các số có thể (00-99)
        all_possible_numbers = [f"{i:02d}" for i in range(100)]
        
        # Điền các số thiếu vào Counter
        for num in all_possible_numbers:
            if num not in self.number_frequency:
                self.number_frequency[num] = 0
        
        # Thêm các thông tin bổ sung
        processed_df['weekday'] = processed_df['date'].dt.dayofweek
        processed_df['day'] = processed_df['date'].dt.day
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['quarter'] = processed_df['date'].dt.quarter
        
        # Tạo các đặc trưng về độ trễ của các số
        for i in range(100):
            num_str = f"{i:02d}"
            processed_df[f'days_since_{num_str}'] = 0
            last_seen = None
            
            for idx, row in processed_df.iterrows():
                if last_seen is not None:
                    days = (row['date'] - last_seen).days
                    processed_df.at[idx, f'days_since_{num_str}'] = days
                
                # Kiểm tra xem số này có xuất hiện trong kỳ quay hiện tại không
                appeared = False
                for col in [f"{col}_last2" for col in self.prize_columns]:
                    if row[col] == num_str:
                        appeared = True
                        last_seen = row['date']
                        break
        
        self.processed_data = processed_df
        print("✅ Đã xử lý dữ liệu thành công.")
        
        # Phân tích các cặp số hay xuất hiện cùng nhau
        self._analyze_combinations()
        
        return self.processed_data
    
    def _analyze_combinations(self):
        """
        Phân tích các cặp số xuất hiện cùng nhau trong một kỳ quay.
        """
        print("\n🔍 Đang phân tích các cặp số hay xuất hiện cùng nhau...")
        combinations = Counter()
        
        for idx, row in tqdm(self.processed_data.iterrows(), total=len(self.processed_data)):
            # Lấy tất cả các số trong kỳ quay này
            numbers = []
            for col in [f"{col}_last2" for col in self.prize_columns]:
                numbers.append(row[col])
            
            # Tạo tất cả các cặp có thể
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    combinations[pair] += 1
        
        self.combinations = combinations
        print(f"✅ Đã phân tích {len(combinations)} cặp số.")
    
    def get_hot_numbers(self, n=10, window=None):
        """
        Lấy các số nóng (hot numbers) - các số xuất hiện nhiều nhất.
        
        Args:
            n (int): Số lượng số nóng cần lấy.
            window (int, optional): Chỉ xét trong window kỳ gần nhất.
            
        Returns:
            list: Danh sách n số nóng.
        """
        if window:
            # Chỉ xét window kỳ gần nhất
            recent_data = self.processed_data.iloc[-window:]
            all_numbers = []
            for col in [f"{col}_last2" for col in self.prize_columns]:
                all_numbers.extend(recent_data[col].tolist())
            counter = Counter(all_numbers)
        else:
            counter = self.number_frequency
        
        return counter.most_common(n)
    
    def get_cold_numbers(self, n=10, min_days=30):
        """
        Lấy các số lạnh (cold numbers) - các số không xuất hiện trong thời gian dài.
        
        Args:
            n (int): Số lượng số lạnh cần lấy.
            min_days (int): Số ngày tối thiểu không xuất hiện.
            
        Returns:
            list: Danh sách n số lạnh.
        """
        last_date = self.processed_data['date'].max()
        cold_numbers = []
        
        # Kiểm tra từng số
        for i in range(100):
            num_str = f"{i:02d}"
            appeared = False
            
            # Kiểm tra trong các kỳ quay gần đây
            for idx in range(len(self.processed_data)-1, max(0, len(self.processed_data)-min_days), -1):
                row = self.processed_data.iloc[idx]
                
                for col in [f"{col}_last2" for col in self.prize_columns]:
                    if row[col] == num_str:
                        appeared = True
                        days_ago = (last_date - row['date']).days
                        cold_numbers.append((num_str, days_ago))
                        break
                
                if appeared:
                    break
            
            # Nếu không xuất hiện trong min_days ngày gần đây
            if not appeared:
                # Tìm lần cuối xuất hiện
                for idx in range(len(self.processed_data)-min_days-1, -1, -1):
                    row = self.processed_data.iloc[idx]
                    for col in [f"{col}_last2" for col in self.prize_columns]:
                        if row[col] == num_str:
                            days_ago = (last_date - row['date']).days
                            cold_numbers.append((num_str, days_ago))
                            appeared = True
                            break
                    if appeared:
                        break
                
                # Nếu chưa từng xuất hiện
                if not appeared:
                    cold_numbers.append((num_str, float('inf')))
        
        # Sắp xếp theo số ngày không xuất hiện (từ lâu đến gần)
        cold_numbers.sort(key=lambda x: x[1], reverse=True)
        return cold_numbers[:n]
    
    def get_hot_pairs(self, n=10, window=None):
        """
        Lấy các cặp số hay xuất hiện cùng nhau.
        
        Args:
            n (int): Số lượng cặp cần lấy.
            window (int, optional): Chỉ xét trong window kỳ gần nhất.
            
        Returns:
            list: Danh sách n cặp số hay xuất hiện cùng nhau.
        """
        if window:
            # Phân tích lại các cặp số trong window kỳ gần nhất
            combinations = Counter()
            recent_data = self.processed_data.iloc[-window:]
            
            for idx, row in recent_data.iterrows():
                # Lấy tất cả các số trong kỳ quay này
                numbers = []
                for col in [f"{col}_last2" for col in self.prize_columns]:
                    numbers.append(row[col])
                
                # Tạo tất cả các cặp có thể
                for i in range(len(numbers)):
                    for j in range(i+1, len(numbers)):
                        pair = tuple(sorted([numbers[i], numbers[j]]))
                        combinations[pair] += 1
            
            return combinations.most_common(n)
        else:
            return self.combinations.most_common(n)
    
    def analyze_consecutive_patterns(self):
        """
        Phân tích các mẫu số liên tiếp (ví dụ: 23, 24, 25).
        
        Returns:
            Counter: Đếm các mẫu số liên tiếp.
        """
        consecutive_patterns = Counter()
        
        for idx, row in self.processed_data.iterrows():
            # Lấy tất cả các số trong kỳ quay này
            numbers = []
            for col in [f"{col}_last2" for col in self.prize_columns]:
                numbers.append(int(row[col]))
            
            # Sắp xếp các số
            numbers.sort()
            
            # Tìm các mẫu liên tiếp
            consecutive = []
            for i in range(len(numbers) - 1):
                if numbers[i+1] == numbers[i] + 1:
                    if not consecutive or consecutive[-1] != numbers[i]:
                        consecutive.append(numbers[i])
                    consecutive.append(numbers[i+1])
                elif consecutive:
                    if len(consecutive) >= 3:
                        pattern = tuple(consecutive)
                        consecutive_patterns[pattern] += 1
                    consecutive = []
            
            # Kiểm tra mẫu liên tiếp cuối cùng
            if consecutive and len(consecutive) >= 3:
                pattern = tuple(consecutive)
                consecutive_patterns[pattern] += 1
        
        return consecutive_patterns

    def analyze_number_distribution(self):
        """
        Phân tích phân phối của các số.
        
        Returns:
            dict: Thông tin về phân phối các số.
        """
        distribution = {}
        
        # Phân tích phân phối theo chữ số đầu và chữ số cuối
        first_digit = Counter()
        second_digit = Counter()
        
        for num, count in self.number_frequency.items():
            first_digit[num[0]] += count
            second_digit[num[1]] += count
        
        distribution['first_digit'] = first_digit
        distribution['second_digit'] = second_digit
        
        # Phân tích phân phối theo chẵn/lẻ
        even_odd = {'even': 0, 'odd': 0, 'mixed': 0}
        
        for num, count in self.number_frequency.items():
            if int(num[0]) % 2 == 0 and int(num[1]) % 2 == 0:
                even_odd['even'] += count
            elif int(num[0]) % 2 != 0 and int(num[1]) % 2 != 0:
                even_odd['odd'] += count
            else:
                even_odd['mixed'] += count
        
        distribution['even_odd'] = even_odd
        
        # Phân tích phân phối theo tổng chữ số
        sum_digits = Counter()
        
        for num, count in self.number_frequency.items():
            digit_sum = int(num[0]) + int(num[1])
            sum_digits[digit_sum] += count
        
        distribution['sum_digits'] = sum_digits
        
        return distribution
    
    def visualize_number_frequency(self, top_n=20):
        """
        Trực quan hóa tần suất xuất hiện của các số.
        
        Args:
            top_n (int): Số lượng số hiển thị.
        """
        plt.figure(figsize=(12, 8))
        
        # Top n số có tần suất cao nhất
        most_common = self.number_frequency.most_common(top_n)
        numbers = [x[0] for x in most_common]
        counts = [x[1] for x in most_common]
        
        bars = plt.bar(numbers, counts, color='royalblue')
        plt.xlabel('Số', fontsize=12)
        plt.ylabel('Tần suất xuất hiện', fontsize=12)
        plt.title(f'Top {top_n} số có tần suất xuất hiện cao nhất', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_trends(self, numbers, window=50):
        """
        Trực quan hóa xu hướng của các số theo thời gian.
        
        Args:
            numbers (list): Danh sách các số cần phân tích.
            window (int): Kích thước cửa sổ trượt.
        """
        plt.figure(figsize=(15, 8))
        
        for num in numbers:
            # Tạo mảng đánh dấu xuất hiện của số
            appearances = []
            for idx, row in self.processed_data.iterrows():
                # Kiểm tra xem số này có xuất hiện không
                appeared = False
                for col in [f"{col}_last2" for col in self.prize_columns]:
                    if row[col] == num:
                        appeared = True
                        break
                appearances.append(1 if appeared else 0)
            
            # Tính trung bình trượt
            rolling_avg = pd.Series(appearances).rolling(window=window).mean()
            
            # Vẽ đồ thị
            plt.plot(self.processed_data['date'][-len(rolling_avg):], 
                    rolling_avg, 
                    label=f'Số {num}')
        
        plt.xlabel('Ngày', fontsize=12)
        plt.ylabel(f'Tần suất trung bình (cửa sổ {window} kỳ)', fontsize=12)
        plt.title('Xu hướng xuất hiện của các số theo thời gian', fontsize=14)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def create_features(self, window_size=10, sequence_length=5):
        """
        Tạo đặc trưng cho mô hình học máy.
        
        Args:
            window_size (int): Số kỳ quay trước để tính tần suất.
            sequence_length (int): Độ dài chuỗi cho LSTM.
            
        Returns:
            tuple: (X_freq, y_freq) - Đặc trưng và nhãn cho mô hình tần suất,
                  (X_seq, y_seq) - Đặc trưng và nhãn cho mô hình chuỗi.
        """
        print("\n🔧 Đang tạo đặc trưng cho mô hình học máy...")
        
        # Tạo đặc trưng dựa trên tần suất
        X_freq = []
        y_freq = []
        
        # Tạo danh sách các cột 2 số cuối
        last2_columns = [f"{col}_last2" for col in self.prize_columns]
        
        # Vòng lặp qua từng kỳ quay
        for i in tqdm(range(window_size, len(self.processed_data)), desc="Đặc trưng tần suất"):
            # Lấy dữ liệu trong cửa sổ
            window_data = self.processed_data.iloc[i-window_size:i]
            
            # 1. Tần suất xuất hiện của mỗi số trong cửa sổ
            window_numbers = []
            for _, row in window_data.iterrows():
                for col in last2_columns:
                    window_numbers.append(row[col])
            
            window_freq = Counter(window_numbers)
            
            # 2. Thông tin thời gian
            current_date = self.processed_data.iloc[i]['date']
            time_features = [
                current_date.dayofweek,  # Thứ trong tuần
                current_date.day,        # Ngày trong tháng
                current_date.month,      # Tháng
                current_date.quarter     # Quý
            ]
            
            # 3. Độ trễ của các số
            delay_features = []
            for j in range(100):
                num_str = f"{j:02d}"
                delay_features.append(self.processed_data.iloc[i][f'days_since_{num_str}'])
            
            # 4. Tạo vector đặc trưng
            feature_vector = []
            
            # Thêm tần suất của mỗi số
            for j in range(100):
                num_str = f"{j:02d}"
                freq = window_freq.get(num_str, 0)
                feature_vector.append(freq)
            
            # Thêm đặc trưng thời gian
            feature_vector.extend(time_features)
            
            # Thêm đặc trưng độ trễ
            feature_vector.extend(delay_features)
            
            # Thêm vào danh sách đặc trưng
            X_freq.append(feature_vector)
            
            # Tạo nhãn từ kỳ quay hiện tại
            current_numbers = []
            for col in last2_columns:
                current_numbers.append(self.processed_data.iloc[i][col])
            
            y_freq.append(current_numbers)
        
        # Tạo đặc trưng chuỗi cho LSTM (nếu có)
        X_seq = []
        y_seq = []
        
        if PYTORCH_AVAILABLE:
            # Tạo dữ liệu chuỗi cho LSTM
            for i in tqdm(range(sequence_length, len(self.processed_data)), desc="Đặc trưng chuỗi"):
                # Tạo chuỗi các kỳ quay trước
                sequence = []
                
                for j in range(i - sequence_length, i):
                    # Tạo vector cho mỗi kỳ quay
                    draw_vector = []
                    
                    # Thêm các số trong kỳ quay
                    for col in last2_columns:
                        num = int(self.processed_data.iloc[j][col])
                        draw_vector.append(num)
                    
                    # Thêm đặc trưng thời gian
                    draw_date = self.processed_data.iloc[j]['date']
                    draw_vector.extend([
                        draw_date.dayofweek,
                        draw_date.day,
                        draw_date.month
                    ])
                    
                    sequence.append(draw_vector)
                
                # Thêm chuỗi vào danh sách đặc trưng
                X_seq.append(sequence)
                
                # Tạo nhãn từ kỳ quay hiện tại
                current_numbers = []
                for col in last2_columns:
                    current_numbers.append(int(self.processed_data.iloc[i][col]))
                
                # Tạo vector nhãn one-hot
                y_vector = np.zeros(100)
                for num in current_numbers:
                    y_vector[num] = 1
                
                y_seq.append(y_vector)
            
            # Chuyển sang numpy array
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
        
        # Chuyển sang numpy array
        X_freq = np.array(X_freq)
        y_freq = np.array(y_freq)
        
        print(f"✅ Đã tạo {len(X_freq)} mẫu đặc trưng cho mô hình tần suất.")
        if PYTORCH_AVAILABLE:
            print(f"✅ Đã tạo {len(X_seq)} mẫu đặc trưng cho mô hình chuỗi.")
        
        return (X_freq, y_freq), (X_seq, y_seq) if PYTORCH_AVAILABLE else (None, None)

class LotteryPredictor:
    """
    Lớp dự đoán kết quả xổ số sử dụng nhiều phương pháp.
    """
    
    def __init__(self, analyzer):
        """
        Khởi tạo đối tượng LotteryPredictor.
        
        Args:
            analyzer (LotteryAnalyzer): Đối tượng phân tích dữ liệu.
        """
        self.analyzer = analyzer
        self.frequency_model = None
        self.ml_model = None
        self.lstm_model = None
        self.ensemble_model = None
        self.scaler = None
        self.models_dir = "models"
        
        # Tạo thư mục lưu mô hình nếu chưa tồn tại
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def predict_by_frequency(self, top_n=10, window=None):
        """
        Dự đoán dựa trên phương pháp tần suất.
        
        Args:
            top_n (int): Số lượng số dự đoán.
            window (int, optional): Chỉ xét trong window kỳ gần nhất.
            
        Returns:
            list: Danh sách top_n số có tần suất cao nhất.
        """
        print("\n📊 Dự đoán bằng phương pháp tần suất...")
        
        # Lấy các số nóng
        hot_numbers = self.analyzer.get_hot_numbers(top_n, window)
        return hot_numbers
    
    def predict_by_pattern(self, top_n=10):
        """
        Dự đoán dựa trên các mẫu và xu hướng.
        
        Args:
            top_n (int): Số lượng số dự đoán.
            
        Returns:
            list: Danh sách số dự đoán dựa trên mẫu.
        """
        print("\n🔍 Dự đoán bằng phương pháp phân tích mẫu...")
        
        predictions = []
        
        # 1. Kết hợp các số nóng và lạnh
        hot_numbers = self.analyzer.get_hot_numbers(top_n // 2, window=20)
        cold_numbers = self.analyzer.get_cold_numbers(top_n // 2, min_days=30)
        
        # Thêm vào danh sách dự đoán
        for num, count in hot_numbers:
            predictions.append((num, count, 'hot'))
        
        for num, days in cold_numbers:
            # Ưu tiên thấp hơn so với số nóng
            weight = 1000 - days if days < 1000 else 0
            predictions.append((num, weight, 'cold'))
        
        # 2. Phân tích các cặp số hay xuất hiện cùng nhau
        hot_pairs = self.analyzer.get_hot_pairs(top_n, window=30)
        
        for pair, count in hot_pairs:
            # Tìm xem số nào từ cặp đã có trong dự đoán
            for num in pair:
                if any(num == pred[0] for pred in predictions):
                    # Thêm số còn lại vào dự đoán
                    other_num = pair[1] if num == pair[0] else pair[0]
                    if not any(other_num == pred[0] for pred in predictions):
                        predictions.append((other_num, count // 2, 'pair'))
        
        # 3. Dựa vào phân phối số
        distribution = self.analyzer.analyze_number_distribution()
        
        # Tìm chữ số đầu và chữ số cuối phổ biến
        top_first_digits = sorted(distribution['first_digit'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_second_digits = sorted(distribution['second_digit'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Tạo số dựa trên các chữ số phổ biến
        for first in [d[0] for d in top_first_digits]:
            for second in [d[0] for d in top_second_digits]:
                num = first + second
                # Kiểm tra xem số này đã có trong dự đoán chưa
                if not any(num == pred[0] for pred in predictions):
                    weight = (distribution['first_digit'][first] + distribution['second_digit'][second]) // 2
                    predictions.append((num, weight, 'distribution'))
        
        # Sắp xếp theo trọng số giảm dần và lấy top_n
        predictions.sort(key=lambda x: x[1], reverse=True)
        result = [(num, weight) for num, weight, _ in predictions[:top_n]]
        
        return result
    
    def train_machine_learning_model(self, save_model=True):
        """
        Huấn luyện mô hình học máy để dự đoán.
        
        Args:
            save_model (bool): Có lưu mô hình sau khi huấn luyện không.
            
        Returns:
            tuple: (model, accuracy) - Mô hình đã huấn luyện và độ chính xác.
        """
        print("\n🤖 Huấn luyện mô hình học máy...")
        
        # Tạo đặc trưng cho mô hình
        (X_freq, y_freq), (X_seq, y_seq) = self.analyzer.create_features()
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X_freq, y_freq, test_size=0.2, random_state=42)
        
        # Chuẩn hóa đặc trưng
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        
        # Làm phẳng mảng nhãn
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
        
        # Nhân rộng đặc trưng tương ứng
        X_train_repeated = np.repeat(X_train_scaled, y_train.shape[1], axis=0)
        X_test_repeated = np.repeat(X_test_scaled, y_test.shape[1], axis=0)
        
        # Pipeline cho GridSearchCV
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=100)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Tham số cho GridSearchCV
        param_grid = {
            'feature_selection__k': [50, 100, 150],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None]
        }
        
        # GridSearchCV
        print("🔍 Tìm kiếm tham số tối ưu...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_repeated, y_train_flat)
        
        print(f"✅ Tham số tối ưu: {grid_search.best_params_}")
        
        # Huấn luyện mô hình tốt nhất
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_repeated, y_train_flat)
        
        # Đánh giá mô hình
        y_pred = best_model.predict(X_test_repeated)
        accuracy = accuracy_score(y_test_flat, y_pred)
        precision = precision_score(y_test_flat, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_flat, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_flat, y_pred, average='macro', zero_division=0)
        
        print(f"📊 Hiệu suất mô hình:")
        print(f"  - Độ chính xác (Accuracy): {accuracy*100:.2f}%")
        print(f"  - Độ chính xác (Precision): {precision*100:.2f}%")
        print(f"  - Độ nhạy (Recall): {recall*100:.2f}%")
        print(f"  - F1-score: {f1*100:.2f}%")
        
        # Lưu mô hình nếu cần
        if save_model:
            model_path = os.path.join(self.models_dir, "ml_model.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"💾 Đã lưu mô hình tại {model_path}")
        
        self.ml_model = best_model
        return best_model, accuracy
        
    def train_lstm_model(self, save_model=True):
        """
        Huấn luyện mô hình LSTM để dự đoán.
        
        Args:
            save_model (bool): Có lưu mô hình sau khi huấn luyện không.
            
        Returns:
            tuple: (model, accuracy) - Mô hình đã huấn luyện và độ chính xác.
        """
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch không khả dụng, không thể huấn luyện mô hình LSTM.")
            return None, 0
        
        print("\n🧠 Huấn luyện mô hình LSTM...")
        
        # Tạo đặc trưng chuỗi cho LSTM
        _, (X_seq, y_seq) = self.analyzer.create_features()
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        
        # Chuyển sang tensor PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Tạo dataset và dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        
        # Tham số mô hình
        input_dim = X_train.shape[2]  # Số đặc trưng đầu vào
        hidden_dim = 128  # Kích thước lớp ẩn
        layer_dim = 2    # Số lớp LSTM
        output_dim = 100  # Số lớp đầu ra (100 số từ 00-99)
        dropout_prob = 0.2  # Tỉ lệ dropout
        
        # Khởi tạo mô hình
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
        
        # Hàm mất mát và optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Tham số huấn luyện
        num_epochs = 50
        best_loss = float('inf')
        patience = 10
        counter = 0
        best_model_state = None
        
        # Huấn luyện mô hình
        print("🏋️‍♂️ Bắt đầu huấn luyện LSTM...")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            # Vòng lặp huấn luyện
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward và optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Tính loss trung bình
            train_loss = train_loss / len(train_loader)
            
            # Đánh giá trên tập kiểm tra
            model.eval()
            test_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            test_loss = test_loss / len(test_loader)
            
            # In thông tin
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}')
            
            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping tại epoch {epoch+1}')
                    break
        
        # Khôi phục mô hình tốt nhất
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Đánh giá mô hình
        model.eval()
        y_pred_list = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                y_pred = model(batch_X)
                y_pred_list.append(y_pred.numpy())
        
        y_pred = np.vstack(y_pred_list)
        
        # Đánh giá theo độ chính xác của top k dự đoán
        accuracy = 0.0
        for i in range(len(y_test)):
            true_indices = np.where(y_test[i] > 0.5)[0]
            top_k_pred = np.argsort(y_pred[i])[::-1][:len(true_indices)]
            correct = len(set(true_indices) & set(top_k_pred))
            accuracy += correct / len(true_indices) if len(true_indices) > 0 else 0
        
        accuracy = accuracy / len(y_test) * 100
        print(f"📊 Hiệu suất mô hình LSTM:")
        print(f"  - Độ chính xác: {accuracy:.2f}%")
        print(f"  - Loss: {best_loss:.4f}")
        
        # Lưu mô hình nếu cần
        if save_model:
            model_path = os.path.join(self.models_dir, "lstm_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Đã lưu mô hình LSTM tại {model_path}")
        
        self.lstm_model = model
        return model, accuracy

    def create_ensemble_model(self, save_model=True):
        """
        Tạo mô hình tổng hợp từ nhiều mô hình khác nhau.
        
        Args:
            save_model (bool): Có lưu mô hình sau khi huấn luyện không.
            
        Returns:
            tuple: (model, accuracy) - Mô hình tổng hợp và độ chính xác.
        """
        print("\n🔄 Tạo mô hình tổng hợp (Ensemble)...")
        
        # Tạo đặc trưng
        (X_freq, y_freq), _ = self.analyzer.create_features()
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X_freq, y_freq, test_size=0.2, random_state=42)
        
        # Chuẩn hóa đặc trưng
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Làm phẳng mảng nhãn
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
        
        # Nhân rộng đặc trưng
        X_train_repeated = np.repeat(X_train_scaled, y_train.shape[1], axis=0)
        X_test_repeated = np.repeat(X_test_scaled, y_test.shape[1], axis=0)
        
        # Tạo các mô hình con
        models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ]
        
        # Tạo mô hình tổng hợp
        ensemble = VotingClassifier(estimators=models, voting='soft')
        
        # Huấn luyện mô hình
        print("🏋️‍♂️ Huấn luyện mô hình tổng hợp...")
        ensemble.fit(X_train_repeated, y_train_flat)
        
        # Đánh giá mô hình
        y_pred = ensemble.predict(X_test_repeated)
        accuracy = accuracy_score(y_test_flat, y_pred)
        precision = precision_score(y_test_flat, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_flat, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_flat, y_pred, average='macro', zero_division=0)
        
        print(f"📊 Hiệu suất mô hình tổng hợp:")
        print(f"  - Độ chính xác (Accuracy): {accuracy*100:.2f}%")
        print(f"  - Độ chính xác (Precision): {precision*100:.2f}%")
        print(f"  - Độ nhạy (Recall): {recall*100:.2f}%")
        print(f"  - F1-score: {f1*100:.2f}%")
        
        # Lưu mô hình nếu cần
        if save_model:
            model_path = os.path.join(self.models_dir, "ensemble_model.pkl")
            scaler_path = os.path.join(self.models_dir, "ensemble_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"💾 Đã lưu mô hình tổng hợp tại {model_path}")
        
        self.ensemble_model = ensemble
        self.scaler = scaler
        return ensemble, accuracy
    
    def load_models(self):
        """
        Tải các mô hình đã lưu.
        
        Returns:
            bool: True nếu tải thành công, False nếu không.
        """
        print("\n📂 Đang tải các mô hình đã lưu...")
        
        try:
            # Tải mô hình học máy
            ml_model_path = os.path.join(self.models_dir, "ml_model.pkl")
            if os.path.exists(ml_model_path):
                with open(ml_model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                print(f"✅ Đã tải mô hình học máy từ {ml_model_path}")
            
            # Tải mô hình tổng hợp
            ensemble_model_path = os.path.join(self.models_dir, "ensemble_model.pkl")
            if os.path.exists(ensemble_model_path):
                with open(ensemble_model_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
                print(f"✅ Đã tải mô hình tổng hợp từ {ensemble_model_path}")
            
            # Tải bộ chuẩn hóa
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Đã tải bộ chuẩn hóa từ {scaler_path}")
            
            # Tải mô hình LSTM
            lstm_model_path = os.path.join(self.models_dir, "lstm_model.pt")
            if PYTORCH_AVAILABLE and os.path.exists(lstm_model_path):
                # Khởi tạo mô hình với cấu trúc giống như khi huấn luyện
                input_dim = 0  # Sẽ được cập nhật sau
                hidden_dim = 128
                layer_dim = 2
                output_dim = 100
                dropout_prob = 0.2
                
                # Tính input_dim từ dữ liệu
                _, (X_seq, _) = self.analyzer.create_features(sequence_length=5)
                if X_seq is not None:
                    input_dim = X_seq.shape[2]
                    
                    # Khởi tạo mô hình
                    self.lstm_model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
                    
                    # Tải tham số
                    self.lstm_model.load_state_dict(torch.load(lstm_model_path))
                    self.lstm_model.eval()  # Chuyển sang chế độ đánh giá
                    
                    print(f"✅ Đã tải mô hình LSTM từ {lstm_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            return False
    
    def predict_by_machine_learning(self, top_n=10):
        """
        Dự đoán bằng mô hình học máy.
        
        Args:
            top_n (int): Số lượng số cần dự đoán.
            
        Returns:
            list: Danh sách (số, xác suất) theo dự đoán của mô hình học máy.
        """
        print("\n🤖 Dự đoán bằng mô hình học máy...")
        
        if self.ml_model is None:
            print("❌ Chưa có mô hình học máy. Vui lòng gọi phương thức train_machine_learning_model() trước.")
            return None
        
        # Lấy dữ liệu mới nhất
        window_size = 10  # Kích thước cửa sổ mặc định
        
        # Lấy dữ liệu trong cửa sổ
        latest_data = self.analyzer.processed_data.iloc[-window_size:]
        
        # Tạo đặc trưng tương tự như trong train_machine_learning_model
        # 1. Tần suất xuất hiện
        last2_columns = [f"{col}_last2" for col in self.analyzer.prize_columns]
        window_numbers = []
        for _, row in latest_data.iterrows():
            for col in last2_columns:
                window_numbers.append(row[col])
        
        window_freq = Counter(window_numbers)
        
        # 2. Thông tin thời gian
        current_date = self.analyzer.processed_data.iloc[-1]['date'] + timedelta(days=7)  # Dự đoán cho kỳ tiếp theo
        time_features = [
            current_date.dayofweek,
            current_date.day,
            current_date.month,
            current_date.quarter
        ]
        
        # 3. Độ trễ của các số
        delay_features = []
        for j in range(100):
            num_str = f"{j:02d}"
            delay_features.append(self.analyzer.processed_data.iloc[-1][f'days_since_{num_str}'])
        
        # 4. Tạo vector đặc trưng
        feature_vector = []
        
        # Thêm tần suất của mỗi số
        for j in range(100):
            num_str = f"{j:02d}"
            freq = window_freq.get(num_str, 0)
            feature_vector.append(freq)
        
        # Thêm đặc trưng thời gian
        feature_vector.extend(time_features)
        
        # Thêm đặc trưng độ trễ
        feature_vector.extend(delay_features)
        
        # Chuyển thành numpy array và chuẩn hóa
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Dự đoán xác suất
        probabilities = self.ml_model.predict_proba(X_scaled)[0]
        
        # Tạo danh sách (số, xác suất)
        number_probs = []
        for i, prob in enumerate(probabilities):
            if i < len(self.ml_model.classes_) and self.ml_model.classes_[i] < 100:
                class_idx = self.ml_model.classes_[i]
                number = f"{class_idx:02d}"
                number_probs.append((number, prob))
        
        # Sắp xếp theo xác suất giảm dần và lấy top_n
        number_probs.sort(key=lambda x: x[1], reverse=True)
        return number_probs[:top_n]

# Định nghĩa lớp mô hình LSTM với PyTorch
class LSTMModel(nn.Module):
    """
    Mô hình LSTM sử dụng PyTorch.
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """
        Khởi tạo mô hình LSTM.
        
        Args:
            input_dim (int): Kích thước đầu vào.
            hidden_dim (int): Kích thước lớp ẩn.
            layer_dim (int): Số lớp LSTM.
            output_dim (int): Kích thước đầu ra.
            dropout_prob (float): Xác suất dropout.
        """
        super(LSTMModel, self).__init__()
        
        # Định nghĩa các tham số
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # Lớp LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout_prob
        )
        
        # Lớp normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Lớp dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Lớp fully connected đầu ra
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Hàm kích hoạt
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Truyền dữ liệu qua mô hình.
        
        Args:
            x (torch.Tensor): Dữ liệu đầu vào có kích thước (batch_size, seq_len, input_dim).
            
        Returns:
            torch.Tensor: Giá trị dự đoán.
        """
        # Khởi tạo trạng thái ẩn
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # Truyền qua LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Lấy đầu ra từ bước cuối cùng
        out = out[:, -1, :]
        
        # Chuẩn hóa batch
        out = self.batch_norm(out)
        
        # Đi qua lớp fully connected
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Áp dụng sigmoid cho đầu ra
        out = self.sigmoid(out)
        
        return out