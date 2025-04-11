#!/usr/bin/env python
"""
修复缩进错误的脚本
"""

import re

# 读取文件内容
with open('scripts/grid_search_ui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复第一处错误 - if not self.custom_grid_params: 后面的for循环需要正确缩进
pattern = r'if not self.custom_grid_params:\s+for special_combo in SPECIAL_COMBINATIONS:'
replacement = 'if not self.custom_grid_params:\n                for special_combo in SPECIAL_COMBINATIONS:'
content = re.sub(pattern, replacement, content)

# 修复try-with块的缩进问题
pattern = r'try:\s+with mp\.Pool\(processes=num_processes\) as pool:'
replacement = 'try:\n                    with mp.Pool(processes=num_processes) as pool:'
content = re.sub(pattern, replacement, content)

# 修复在try-with块中的缩进问题
pattern = r'# 增强版imap处理，添加超时检查\s+result_iter = pool\.imap\(process_param_search, batch_args\)'
replacement = '                        # 增强版imap处理，添加超时检查\n                        result_iter = pool.imap(process_param_search, batch_args)'
content = re.sub(pattern, replacement, content)

# 修复if not self.running:的缩进
pattern = r'# 频繁检查是否应该停止\s+if not self\.running:'
replacement = '                            # 频繁检查是否应该停止\n                            if not self.running:'
content = re.sub(pattern, replacement, content)

# 修复if not self.running:后面的缩进
pattern = r'pool\.terminate\(\)  # 立即终止所有进程\s+pool\.join\(0\.5\)'
replacement = '                                pool.terminate()  # 立即终止所有进程\n                                pool.join(0.5)'
content = re.sub(pattern, replacement, content)

# 修复处理结果部分缩进
pattern = r'# 处理结果\s+if result is not None:'
replacement = '                            # 处理结果\n                            if result is not None:'
content = re.sub(pattern, replacement, content)

# 修复try块和except块的缩进
pattern = r'# 实时保存结果\s+try:\s+with open\(self\.searcher\.results_file, \'wb\'\) as f:'
replacement = '                                # 实时保存结果\n                                try:\n                                    with open(self.searcher.results_file, \'wb\') as f:'
content = re.sub(pattern, replacement, content)

pattern = r'pickle\.dump\(results, f\)\s+except StopIteration:'
replacement = '                                        pickle.dump(results, f)\n                                except StopIteration:'
content = re.sub(pattern, replacement, content)

pattern = r'# 所有结果都已处理完\s+break\s+except Exception as e:'
replacement = '                                    # 所有结果都已处理完\n                                    break\n                                except Exception as e:'
content = re.sub(pattern, replacement, content)

# 修复文件末尾if self.running:的缩进
pattern = r'# 所有搜索结束后，生成一次最终报告\s+if self\.running:'
replacement = '            # 所有搜索结束后，生成一次最终报告\n            if self.running:'
content = re.sub(pattern, replacement, content)

# 写回文件
with open('scripts/grid_search_ui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("已修复scripts/grid_search_ui.py中的缩进错误") 