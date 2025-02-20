# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:16:18 2025

@author: jperezr
"""

import streamlit as st
import spam_detector as sd
import os

# Título de la aplicación
st.title("Sistema de Detección de Spam")

# Sidebar para seleccionar la fase (Entrenamiento o Prueba)
phase = st.sidebar.selectbox("Selecciona la fase", ["Entrenamiento", "Prueba"])

if phase == "Entrenamiento":
    st.header("Fase de Entrenamiento")
    
    if st.button("Entrenar Modelo"):
        st.write("Entrenando el modelo...")
        
        # Llamar a las funciones de entrenamiento
        nb_of_allEmails = sd.number_of_allEmails()
        nb_of_spamEmails = sd.number_of_spamEmails()
        nb_of_hamEmails = sd.number_of_hamEmails()
        
        all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
        all_uniqueWords = sd.unique_words(all_trainWords)
        
        spam_bagOfWords, ham_bagOfWords = sd.bagOfWords_generator(all_uniqueWords, spam_trainWords, ham_trainWords)
        smoothed_spamBOW, smoothed_hamBOW = sd.smoothed_bagOfWords(all_uniqueWords, spam_bagOfWords, ham_bagOfWords, 0.5)
        
        spam_prob = sd.spam_probability(nb_of_allEmails, nb_of_spamEmails)
        ham_prob = sd.ham_probability(nb_of_allEmails, nb_of_hamEmails)
        
        spam_condProb = sd.spam_condProbability(all_uniqueWords, spam_bagOfWords, smoothed_spamBOW, 0.5)
        ham_condProb = sd.ham_condProbability(all_uniqueWords, ham_bagOfWords, smoothed_hamBOW, 0.5)
        
        word_numbers = len(all_uniqueWords)
        words = all_uniqueWords
        ham_wf = ham_bagOfWords
        ham_cp = ham_condProb
        spam_wf = spam_bagOfWords
        spam_cp = spam_condProb
        
        model_output = sd.model_output_generator(word_numbers, words, ham_wf, ham_cp, spam_wf, spam_cp)
        sd.modelFileBuilder(model_output)
        
        st.success("Modelo entrenado y guardado en model.txt")

elif phase == "Prueba":
    st.header("Fase de Prueba")
    
    if st.button("Probar Modelo"):
        st.write("Probando el modelo...")
        
        # Llamar a las funciones de prueba
        test_fileNames = sd.get_testFileNames()
        nb_of_allEmails = sd.number_of_allEmails()
        nb_of_spamEmails = sd.number_of_spamEmails()
        nb_of_hamEmails = sd.number_of_hamEmails()
        
        all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
        all_uniqueWords = sd.unique_words(all_trainWords)
        
        spam_bagOfWords, ham_bagOfWords = sd.bagOfWords_generator(all_uniqueWords, spam_trainWords, ham_trainWords)
        smoothed_spamBOW, smoothed_hamBOW = sd.smoothed_bagOfWords(all_uniqueWords, spam_bagOfWords, ham_bagOfWords, 0.5)
        
        spam_prob = sd.spam_probability(nb_of_allEmails, nb_of_spamEmails)
        ham_prob = sd.ham_probability(nb_of_allEmails, nb_of_hamEmails)
        
        spam_condProb = sd.spam_condProbability(all_uniqueWords, spam_bagOfWords, smoothed_spamBOW, 0.5)
        ham_condProb = sd.ham_condProbability(all_uniqueWords, ham_bagOfWords, smoothed_hamBOW, 0.5)
        
        actual_labels = sd.get_actualLabels()
        ham_scores, spam_scores, predicted_labels, decision_labels = sd.score_calculator(all_uniqueWords, spam_prob, ham_prob, spam_condProb, ham_condProb, 0.5)
        
        fileNumbers = len(test_fileNames)
        fileNames = test_fileNames
        actualLabels = actual_labels
        predictedLabels = predicted_labels
        hamScores = ham_scores
        spamScores = spam_scores
        decisionLabels = decision_labels
        
        result_output = sd.result_output_generator(fileNumbers, fileNames, predictedLabels, hamScores, spamScores, actualLabels, decisionLabels)
        sd.resultFileBuilder(result_output)
        
        # Análisis de resultados
        spam_precision = sd.get_spamPrecision(fileNumbers, actualLabels, predictedLabels)
        spam_recall = sd.get_spamRecall(fileNumbers, actualLabels, predictedLabels)
        spam_accuracy = sd.get_spamAccuracy(fileNumbers, actualLabels, predictedLabels)
        spam_fmeasure = sd.get_spamFmeasure(spam_precision, spam_recall)
        
        ham_precision = sd.get_hamPrecision(fileNumbers, actualLabels, predictedLabels)
        ham_recall = sd.get_hamRecall(fileNumbers, actualLabels, predictedLabels)
        ham_accuracy = sd.get_hamAccuracy(fileNumbers, actualLabels, predictedLabels)
        ham_fmeasure = sd.get_hamFmeasure(ham_precision, ham_recall)
        
        evaluation_result_output = sd.evaluation_result(spam_accuracy, spam_precision, spam_recall, spam_fmeasure, ham_accuracy, ham_precision, ham_recall, ham_fmeasure)
        
        spam_tp, spam_tn, spam_fp, spam_fn = sd.spamConfusionParams(fileNumbers, actualLabels, predictedLabels)
        spam_confusionMatrix_output = sd.spam_confusionMatrix(spam_tp, spam_tn, spam_fp, spam_fn)
        
        ham_tp, ham_tn, ham_fp, ham_fn = sd.hamConfusionParams(fileNumbers, actualLabels, predictedLabels)
        ham_confusionMatrix_output = sd.ham_confusionMatrix(ham_tp, ham_tn, ham_fp, ham_fn)
        
        evaluation_output = sd.evaluation_output_generator(evaluation_result_output, spam_confusionMatrix_output, ham_confusionMatrix_output)
        sd.evaluationFileBuilder(evaluation_output)
        
        st.success("Prueba completada y resultados guardados en result.txt y evaluation.txt")
        
        # Mostrar resultados en la aplicación
        st.subheader("Resultados de la Prueba")
        st.write("**Precisión (Spam):**", spam_precision)
        st.write("**Recall (Spam):**", spam_recall)
        st.write("**Accuracy (Spam):**", spam_accuracy)
        st.write("**F1-measure (Spam):**", spam_fmeasure)
        
        st.write("**Precisión (Ham):**", ham_precision)
        st.write("**Recall (Ham):**", ham_recall)
        st.write("**Accuracy (Ham):**", ham_accuracy)
        st.write("**F1-measure (Ham):**", ham_fmeasure)
        
        st.subheader("Matriz de Confusión (Spam)")
        st.text(spam_confusionMatrix_output)
        
        st.subheader("Matriz de Confusión (Ham)")
        st.text(ham_confusionMatrix_output)