FROM python:3.6.10-slim-buster
RUN apt-get update && apt-get install libgomp1
RUN apt-get install sshpass -y
RUN mkdir -p /usr/share/man/man1
RUN apt install default-jdk -y
RUN apt install default-jre -y
WORKDIR /root/
RUN mkdir caa caascript

WORKDIR /root/caa/

WORKDIR /root/caascript/

COPY all.py requirements.txt requirement_riv.txt directory_creation_ui.py /root/caascript/
RUN mkdir rpscript res litmscript 
WORKDIR /root/caascript/res
COPY bg_css.py feature_df.csv /root/caascript/res/
RUN mkdir bg
COPY bg /root/caascript/res/bg


WORKDIR /root/caascript/rpscript/
RUN mkdir rpmodelling rpmonitor rpanalysis

WORKDIR /root/caascript/rpscript/rpmodelling
COPY login.py mod2.py data_prep.py DataExt.py dashdem.py auto_pilot.py putty.py Data_csv.py client_secret.json /root/caascript/rpscript/rpmodelling/
COPY preprocess.py topmodel.py directory_creation.py HistoryGeneration.py JsonCreation.py SubsetsCreation.py FeaturesCreation.py TrainTestSplit.py model_data.csv PMML_creation.py generate_Test_Files.py /root/caascript/rpscript/rpmodelling/
COPY RP_monitoring.py RP_Monitoring_Automated_modified.py Data_csv.py client_secret.json /root/caascript/rpscript/rpmonitor/
COPY RP_Analysis.py rp_analysis_functions.py /root/caascript/rpscript/rpanalysis/


WORKDIR /root/caascript/litmscript
RUN mkdir litmmonitor litmanalysis
WORKDIR /root/caascript/litmscript/litmmonitor
COPY LITM_Monitoring_automation_modified.py Monitor_start.py client_secret.json Data_csv.py /root/caascript/litmscript/litmmonitor/
RUN mkdir sql_query
COPY  sql_clearedCheques.txt sql_failedCheque.txt sql_processedCheque.txt with_intervention.txt /root/caascript/litmscript/litmmonitor/sql_query/

WORKDIR /root/caascript/litmscript/litmanalysis
COPY amount_and_reference_no_capture.py client_secret.json Data_csv.py flow_new.py generate_row_level_data_from_json_updated_labelling.py /root/caascript/litmscript/litmanalysis/
COPY heading_model.py is_remittance.py LITM_Analysis.py LITM_heading.pickle LITM_remittance.pickle LITM_total.pickle remittance_downloader.py s3_path_extraction_ui.py total_model.py /root/caascript/litmscript/litmanalysis/
WORKDIR /root/caascript


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirement_riv.txt
CMD python /root/caascript/directory_creation_ui.py && streamlit run all.py  --server.allowRunOnSave true --server.maxUploadSize=4096
EXPOSE 8501
