# MERL-ICML2024 复现报告 & 项目说明  
GitHub: https://github.com/cheliu-computation/MERL-ICML2024

## 1. 项目简介 
现有ECG自监督学习（eSSL）方法虽能利用无标注信号进行预训练，但因输入级增强易破坏ECG语义、仅关注低层波形模式而忽略高层次临床概念，导致学到的表征缺乏疾病语义、无法零样本分类，且下游任务需大量标注数据才能达到较好性能。本文提出Multimodal ECG Representation Learning（MERL）框架，通过ECG信号与配对临床报告的多模态联合预训练，结合潜空间增强的Uni-Modal Alignment（避免语义破坏）和Cross-Modal Alignment（引入报告中的临床知识监督），使ECG表征天然具备高层次疾病语义；同时在测试阶段提出Clinical Knowledge Enhanced Prompt Engineering（CKEPE），利用LLM从SNOMED-CT等专家验证知识库中提取可靠的疾病亚型与信号特征，动态生成结构化提示，彻底解决了eSSL的语义缺失与提示贫乏问题，实现真正零样本ECG分类，并在6个数据集上平均AUC达75.2%，超越使用10%标注数据线性微调的最佳eSSL方法3.2%。

## 2. 论文核心创新点
1. **提出多模态 ECG 表征学习框架（MERL）**：  
   实现 ECG 信号与临床报告的联合预训练，原生支持零样本分类。关键优势在于：零样本 MERL 性能超越使用 10% 标注数据线性微调的最佳 eSSL 方法；且线性微调后的 MERL 在所有下游数据集和数据比例下，全面优于传统 eSSL 方法。

2. **引入双重对齐机制（CMA + UMA）**：  
   - **跨模态对齐（CMA）**：通过 ECG 特征与报告文本特征的相似性约束，将临床知识注入 ECG 表征；  
   - **单模态对齐（UMA）**：在潜空间对 ECG 特征进行 dropout 增强（而非原始信号级增强），避免破坏信号语义。  
   双重对齐共同提升了表征的临床语义性和鲁棒性。

3. **提出临床知识增强提示工程（CKEPE）**：  
   测试阶段利用 LLM 从专家验证的外部知识库（如 SNOMED-CT）中动态提取疾病亚型与信号特征，生成结构化提示，有效降低大模型“幻觉”问题，显著提升零样本分类的准确性。

## 3. 文章框架
<img src="image_result/framework.png" width="85%" alt="MERL 文章整体框架图">
## 4.核心公式以及对应代码
公式一和公式二：（utils_loss.py）
 ```python
def clip_loss(x, y, temperature=0.07, device='cuda'):
    x = F.normalize(x, dim=-1)  # 嵌入归一化，对应公式中相似度的余弦计算前提
    y = F.normalize(y, dim=-1)  # 正样本标签（自身匹配）

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature  # 计算相似度并除以温度

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels)  # 文本到图像方向的损失
    loss_i = F.cross_entropy(sim.T, labels)  # 图像到文本方向的损失（双向对齐）

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.
    # 最终返回的损失是双向损失的平均，公式一是单方向损失（逻辑一致）
    return (loss_t + loss_i), acc1, acc5
 ```

公式三：（utils_trainer.py）
 ```python
                with autocast():
                    #报告tokenize与模型前向
                    report_tokenize_output = self.model.module._tokenize(report)

                    input_ids = report_tokenize_output.input_ids.to(
                        self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(
                        self.device).contiguous()

                    output_dict = self.model(ecg, input_ids, attention_mask) 
                    ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']

                    #多卡聚合：生成公式三所需要的“同一ECG的两个dropout嵌入”
                    world_size = torch_dist.get_world_size()
                    with torch.no_grad():
                        #初始化聚合列表：存储多卡的ECG dropout嵌入（两个变体）
                        agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]
                        # 多卡聚合两个dropout嵌入（ecg_emb[0]和ecg_emb[1]
                        dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                        dist.all_gather(agg_proj_text_emb, proj_text_emb[0])
                        
                        agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                        dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                        dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                        # get current rank
                        rank = torch_dist.get_rank()
                    # 替换本地数据：确保当前卡的dropout嵌入正确参与计算
                    agg_proj_img_emb[rank] = proj_ecg_emb[0]
                    agg_proj_text_emb[rank] = proj_text_emb[0]
                    # 拼接多卡数据：得到批量的ECG dropout嵌入对
                    agg_proj_ecg_emb1[rank] = ecg_emb[0]
                    agg_proj_ecg_emb2[rank] = ecg_emb[1]

                    agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                    agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                    agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                    agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)
                    # 输入为“同一ECG的两个dropout嵌入”，对应公式三的正样本对
                    cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                    uma_loss, _, _ = clip_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2, device=self.device)
                    loss = cma_loss + uma_loss
 ```
## 5.复现过程
1. 环境配置
   - git 或 download 代码仓库到本地：`https://github.com/cheliu-computation/MERL-ICML2024.git`
   - 环境准备：新建符合 requirements 的 conda 虚拟环境

2. 数据集下载
   - 本文用了四个数据集：一个用于训练（MIMIC-IV-ECG），另外三个用于测试（PTB-XL、CPSC2018、CSN）。本人复现仅下载了 PTB-XL 数据集
   - 数据集的划分：作者提供了相应代码（包括预训练和下游任务数据集的划分）

3. 预训练模型的下载
   - 作者在谷歌云盘上提供了预训练模型，包括用于零样本分类和线性探测的模型

4. 下游任务评估
   - 零样本分类
     ```bash
     cd MERL/zeroshot
     bash zeroshot.sh
   - 线性探测（本人复现的选择了特定的PTB-XL中的rhythm数据集）
    ```bash
     cd MERL/finetune/sub_script
     bash run_all_linear.sh
5. 复现结果（由于数据集过大，因此只做了zeroshot在PTB-XL（resnet18为骨干）和ptb-xl-rhythm的线性探测（VIT-TINY为骨干））

