# Cromwell Metadata Profiler

This project aims to provide a simple and lightweight dashboard to analyze the metadata of Cromwell workflows' metadata.

<p align="center">
  <img src="https://github.com/henriqueribeiro/cromwell_profiler/blob/main/img/profiler.gif" alt="animated" />
</p>

This project is based on Broad Institute gatk-sv [script](https://github.com/broadinstitute/gatk-sv/blob/master/scripts/cromwell/analyze_resource_acquisition.py)

## Features
- High-level statistics (task counts, duration, CPU hours)
- Descriptive statistics about tasks' duration
- Timeline of the entire workflow
- Compute resources used over time

## How to run
### Get metadata file
#### Option 1:
Go to Cromwell server webpage and call `/api/workflows/{version}/{id}/metadata` with the following inputs:
```
id: <workflow_id>
includeKey: id
includeKey: executionStatus
includeKey: backendStatus
includeKey: status
includeKey: callRoot
includeKey: subWorkflowMetadata
includeKey: subWorkflowId
expandSubWorkflows: true
``` 
Save the output as `metadata.json`

#### Option 2:
If you have [cromshell](https://github.com/broadinstitute/cromshell) installed, run:
```bash
cromshell -t100 metadata <workflow_id> > metadata.json
```

### Install dependencies
```
pip install --upgrade streamlit pandas
```

### Launch the dashboard
#### Option 1:
In most cases there is no need to clone the repo. 
```
streamlit run https://raw.githubusercontent.com/henriqueribeiro/cromwell_profiler/main/profiler.py
```

#### Option 2:
```
git clone https://github.com/henriqueribeiro/cromwell_profiler.git
streamlit run profiler.py
```

### Run the profiler
After launching the app, a new page will open on your browser. Just upload the metadata file and the plots will start appearing.

## Troubleshooting
### `File must be 200.0MB or smaller.`

Streamlit sets a maximum file size for uploaded files. If your metadata file is bigger than 200MB do the following:

1. Clone the repo
```
git clone https://github.com/henriqueribeiro/cromwell_profiler.git
```
2. Increase the maximum file size allowed
```
cd cromwell_profiler
mkdir .streamlit
cat <<EOT >> .streamlit/config.toml
[server]
maxUploadSize=1024
EOT
```
In this example we are setting the maximum file size to 1024MB

3. Re-launch the profiler
```
streamlit run profiler.py
```

## Contributing
Please feel free to open PRs with new features, plots, etc

