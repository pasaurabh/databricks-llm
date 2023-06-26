# Databricks notebook source
# MAGIC %sql
# MAGIC --Select distinct(hostname)   from dataai.labelled_validation_data
# MAGIC --select  * from dataai.nvd_cve_csv
# MAGIC select longitude,latitude , location from dataai.host h , dataai.labelled_testing_data td 
# MAGIC where td.hostName = h.hostName 
# MAGIC union 
# MAGIC select longitude,latitude , location from dataai.host h ,dataai.labelled_validation_data vd   
# MAGIC where h.hostName = vd.hostName 
# MAGIC union
# MAGIC select longitude,latitude , location from dataai.host h ,dataai.labelled_training_data_ rgd   
# MAGIC where h.hostName = rgd.hostName 
# MAGIC
