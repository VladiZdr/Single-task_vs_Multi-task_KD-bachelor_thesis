from configs.model_configs import ModelConfig, MultiTaskModelConfig, TfidfBaselineConfig

"----------------------------------------------------------------SINGLE-TASK CONFIGURATIONS---------------------------------------------------------------------------"

ledgar_teacher_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  

    batch_size = 16,

    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_teacher_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  
    
    batch_size=4,    

    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_teacher_low_ressource_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data= 1,
    low_resource_percent=1,  
    
    batch_size=4,    

    unique_id_for_dir = "low_ress_tester",
    preprocessed_data_dir = "raw"
)

ledgar_teacher_low_ressource_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=15,  
    low_resource_percent=50,  

    batch_size = 16,

    unique_id_for_dir = "low_ress_tester",
    preprocessed_data_dir = "raw"
)

ledgar_supervised_student_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,
    batch_size = 16,

    unique_id_for_dir = "supervised_student_tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_supervised_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    
    batch_size=4,
    
    unique_id_for_dir = "supervised_student_tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_check_correct_load_preprocessed_dataset = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    
    batch_size=4,
    
    unique_id_for_dir = "check_correct_load",
    preprocessed_data_dir = "./datasets_store/unfair_tos_preprocessed"
)

unfair_tos_kd_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_tester,

    percent_of_data=1,  
    
    batch_size=4,
    epochs=2,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "kd_student_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs_tester"
)

ledgar_kd_student_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_tester,

    percent_of_data=1,  

    batch_size = 16,
    epochs = 2,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "kd_student_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs_tester"
)

unfair_tos_kd_check_correct_low_ressource = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_tester,

    low_resource_percent=1,  
    
    batch_size=4,
    epochs=2,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "low_ressource_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs_tester"
)

ledgar_kd_check_correct_low_ressource = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_tester,

    low_resource_percent=1,  

    batch_size = 16,
    epochs = 2,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "low_ressource_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs_tester"
)

"----------------------------------------------------------------MULTI-TASK CONFIGURATIONS----------------------------------------------------------------------------"

multi_task_kd_model_tester = MultiTaskModelConfig(
    ledgar_config = ledgar_kd_student_tester,
    unfair_tos_config = unfair_tos_kd_student_tester,
    unique_id_for_dir = "multi_task_kd_model_tester",
)

multi_task_supervised_model_tester = MultiTaskModelConfig(
    ledgar_config = ledgar_supervised_student_tester,
    unfair_tos_config = unfair_tos_supervised_student_tester,
    unique_id_for_dir = "multi_task_supervised_model_tester",
)

multi_task_check_low_resource = MultiTaskModelConfig(
    ledgar_config = ledgar_kd_check_correct_low_ressource,
    unfair_tos_config = unfair_tos_kd_check_correct_low_ressource,
    unique_id_for_dir = "multi_task_low_res_tester"
)

"--------------------------------------------------------------------TF-IDF BASELINE----------------------------------------------------------------------------------"

tfidf_tester = TfidfBaselineConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",

    model_name_or_path="tfidf_baseline", # !!!! important for data processing !!!!

    percent_of_data=1,  
    
    batch_size=4,    

    unique_id_for_dir = "tfidf_tester",
    preprocessed_data_dir = "raw" # irrelevant -> will be overwritten later
)

tfidf_hidden_dim_tester = TfidfBaselineConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",

    model_name_or_path="tfidf_baseline", # !!!! important for data processing !!!!
    hidden_dim = 10000,

    percent_of_data=1,  
    
    batch_size=4,    

    unique_id_for_dir = "tfidf_hidden_dim",
    preprocessed_data_dir = "raw" # irrelevant -> will be overwritten later
)

"------------------------------------------------------------------LISTS OF TEST MODULES----------------------------------------------------------------------------------"

single_task_testers = [
    ledgar_teacher_tester,
    unfair_tos_teacher_tester,
    
    #ledgar_teacher_low_ressource_tester,
    #unfair_tos_teacher_low_ressource_tester,
    unfair_tos_kd_check_correct_low_ressource,
    #ledgar_kd_check_correct_low_ressource,
    
    unfair_tos_supervised_student_tester,
    
    unfair_tos_check_correct_load_preprocessed_dataset,
    
    unfair_tos_kd_student_tester,
    ledgar_kd_student_tester,
]

multi_task_testers = [
    multi_task_kd_model_tester,
    multi_task_supervised_model_tester,
    #multi_task_check_low_resource,
]

tf_idf_testers = [
    tfidf_tester,
    tfidf_hidden_dim_tester
]