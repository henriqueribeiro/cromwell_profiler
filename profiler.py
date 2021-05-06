import json
import logging
import os
from enum import Enum

import dateutil.parser
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

STATUS = Enum(
    "Status",
    [
        "INCOMPLETE",
        "UNRECOGNIZED",
        "FAILED",
        "ABORTED",
        "RUNNING",
        "SUCCEEDED",
    ],
)


def main():
    # Settings for streamlit app
    st.set_page_config(layout="wide", page_title="Cromwell", page_icon=":pig:")

    st.sidebar.title("Cromwell Metadata ")
    st.sidebar.image("https://i.imgur.com/VFTW0Ti.png")

    # Upload metadata file
    metadata_file = st.file_uploader("Choose a file", type="json")
    if metadata_file is not None:
        check_file_nonempty(metadata_file)

        metadata = json.load(metadata_file)

        status, message = validate_workflow(metadata)

        # Sidebar status
        st.sidebar.header(f"Status: {status.name}")

        if status.value <= 2:
            st.error(message)
            st.stop()
        else:
            if message:
                st.warning(message)

        call_metadata = get_call_metadata(metadata)

        if len(call_metadata) == 0:
            st.error("No calls in workflow metadata.")
            st.stop()

        metadata_df, time_df, root_idx, work_idx = process_metadata(
            call_metadata
        )

        ## Sidebar
        st.sidebar.subheader("Options")
        simple_name_bool = st.sidebar.checkbox("Use simple names", value=True)
        seconds_bool = st.sidebar.checkbox("Show time in seconds", value=True)
        work_tasks_bool = st.sidebar.checkbox(
            'Show only "work" tasks', value=True
        )

        idx = (
            work_idx
            if work_tasks_bool
            else pd.Series([True] * len(metadata_df))
        )
        task_name = "task_simple" if simple_name_bool else "task"
        unique_tasks = metadata_df.loc[idx, task_name].unique()

        plt_config = dict(
            {
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            }
        )

        plot_stats(
            metadata_df,
            root_idx,
            work_idx,
            idx,
            task_name,
            unique_tasks,
            plt_config,
        )

        plot_duration(
            metadata_df, idx, task_name, unique_tasks, seconds_bool, plt_config
        )

        plot_timeline(
            metadata_df, time_df, idx, work_idx, task_name, plt_config
        )

        plot_resources(time_df, work_idx, plt_config)


def plot_stats(
    metadata_df, root_idx, work_idx, idx, task_name, unique_tasks, plt_config
):
    # High level stats TODO: styling
    st.header("High Level Stats")
    num_tasks = len(metadata_df)
    num_tasks_work = len(metadata_df[work_idx])
    overall_duration = pd.Timedelta(
        metadata_df[root_idx]["duration_seconds"].values[0],
        unit="seconds",
    )
    cpu_hours = pd.Timedelta(
        metadata_df[work_idx]["duration_seconds"].fillna(0).sum(),
        unit="seconds",
    )

    st.text(f"Number of tasks: {num_tasks}")
    st.text(f'Number of "work" tasks: {num_tasks_work}')
    st.text(f"Overall Duration: {overall_duration}")
    st.text(f"CPU hours: {cpu_hours}")

    # Cache plot
    cached_df = (
        metadata_df.loc[idx]
        .cached.value_counts(dropna=False)
        .reset_index()
        .rename(columns={"cached": "Number of tasks", "index": "Cached"})
    )
    cached_df["Cached"] = cached_df["Cached"].map(
        {True: "True", False: "False", np.NaN: "N/A"}, na_action="ignore"
    )
    cache_fig = px.pie(
        cached_df,
        values="Number of tasks",
        names="Cached",
        color="Cached",
        color_discrete_map={
            "N/A": "#636EFA",
            "True": "#00CC96",
            "False": "#EF553F",
        },
    )
    cache_fig.update_traces(textinfo="label+percent")

    # Disk plot
    disk_df = (
        metadata_df.loc[idx]
        .disk_type.value_counts(dropna=False)
        .reset_index()
        .rename(columns={"disk_type": "Number of tasks", "index": "Disk type"})
    )
    disk_df["Disk type"] = disk_df["Disk type"].map(
        {"HDD": "HDD", "SSD": "SSD", np.NaN: "N/A"}, na_action="ignore"
    )

    disk_fig = px.pie(
        disk_df,
        values="Number of tasks",
        names="Disk type",
        # title="Disk Types",
        color="Disk type",
        color_discrete_map={
            "N/A": "#636EFA",
            "SSD": "#00CC96",
            "HDD": "#EF553F",
        },
    )
    disk_fig.update_traces(textinfo="label+percent")

    col1, col2, col3 = st.beta_columns(3)

    # Task counts
    col1.subheader("Task Count")
    col1.dataframe(
        metadata_df.loc[idx, task_name]
        .value_counts()[unique_tasks]
        .to_frame("#"),
        height=400,
    )

    col2.subheader("Cached Tasks")
    col2.plotly_chart(cache_fig, use_container_width=True, config=plt_config)

    col3.subheader("Disk Types")
    col3.plotly_chart(disk_fig, use_container_width=True, config=plt_config)


def plot_duration(
    metadata_df, idx, task_name, unique_tasks, seconds_bool, plt_config
):
    st.header("Task Duration")

    df = metadata_df.loc[idx]

    # Duration describe dataframe
    duration_df = (
        df.groupby(task_name)["duration_seconds"].describe().loc[unique_tasks]
    )

    duration_df = duration_df[["mean", "min", "25%", "50%", "75%", "max"]]

    null_idx = duration_df.isnull().any(axis=1)

    if null_idx.all():
        st.warning("There are no finished tasks!")
        return
    elif null_idx.any():
        st.warning("Only showing finished tasks!")
        duration_df = duration_df.loc[~null_idx]

    if not seconds_bool:
        duration_df = duration_df.astype("timedelta64[s]").astype("object")

    st.dataframe(duration_df, height=600)

    # Duration plot
    null_idx = df["duration_seconds"].isnull()

    df = df.loc[~null_idx]

    plot_label = {
        task_name: "Task",
        "duration_seconds": "Duration [seconds] - Log Scale",
    }

    range_pts = df["duration_seconds"].max() - df["duration_seconds"].min()
    height = int(min(max(400 + range_pts * 0.03, 400), 2200))

    fig = px.box(
        df,
        x=task_name,
        y="duration_seconds",
        orientation="v",
        points="all",
        labels=plot_label,
        height=height,
        log_y=True,
    )
    st.plotly_chart(fig, use_container_width=True, config=plt_config)


def plot_timeline(metadata_df, time_df, idx, work_idx, task_name, plt_config):

    ## Timeline
    st.header("Timeline")

    metadata_df = metadata_df.loc[idx]

    plot_label = {task_name: "Task"}

    null_idx = (
        metadata_df[["timestamp_start", "timestamp_stop"]].isnull().any(axis=1)
    )

    if null_idx.all():
        st.warning("There are no finished tasks!")
        return
    elif null_idx.any():
        st.warning("Only showing finished tasks!")
        metadata_df = metadata_df.loc[~null_idx]

    n_tasks = metadata_df[task_name].nunique()
    height = int(min(max(n_tasks * 43, 200), 2200))

    fig = px.timeline(
        metadata_df,
        x_start="timestamp_start",
        x_end="timestamp_stop",
        y=task_name,
        color=task_name,
        height=height,
        labels=plot_label,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, config=plt_config)


def plot_resources(time_df, work_idx, plt_config):

    ## Timeline
    st.header("Resources")

    time_df = time_df.loc[work_idx]

    null_idx = time_df["timestamp"].isnull()

    if null_idx.all():
        st.warning("No resources to show!")
        return
    elif null_idx.any():
        st.warning("Only showing tasks with declared resources!")
        time_df = time_df.loc[~null_idx]

    plot_labels = {
        "timestamp": "Timestamp",
        "memory_sum": "RAM [GiB]",
        "cpu_sum": "CPU Cores",
        "disk_size_sum": "Disk Memory [GiB]",
        "count_sum": "VMs",
    }

    # VMs
    fig = px.area(
        time_df.loc[work_idx], x="timestamp", y="count_sum", labels=plot_labels, height=400
    )
    st.plotly_chart(fig, use_container_width=True, config=plt_config)

    # CPU
    fig = px.area(
        time_df.loc[work_idx], x="timestamp", y="cpu_sum", labels=plot_labels, height=400
    )
    st.plotly_chart(fig, use_container_width=True, config=plt_config)

    # RAM
    fig = px.area(
        time_df.loc[work_idx],
        x="timestamp",
        y="memory_sum",
        labels=plot_labels,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, config=plt_config)

    # Disk size
    fig = px.area(
        time_df.loc[work_idx],
        x="timestamp",
        y="disk_size_sum",
        labels=plot_labels,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, config=plt_config)


def check_file_nonempty(f):
    if f.size == 0:
        st.error(f"Metadata input file '{f.name}' is empty.")
        st.stop()


@st.cache
def process_metadata(metadata):
    """
    Based on: https://github.com/broadinstitute/gatk-sv/blob/master/scripts/cromwell/download_monitoring_logs.py
    """

    colnames = [
        "task",
        "timestamp_start",
        "timestamp_stop",
        "cpu",
        "memory",
        "preemptible",
        "disk_size",
        "disk_type",
        "cached",
        "job_id",
        "parent_id",
    ]
    metadata_df = pd.DataFrame(metadata, columns=colnames)

    metadata_df["timestamp_start"] = metadata_df[
        "timestamp_start"
    ].dt.tz_localize(None)
    metadata_df["timestamp_stop"] = metadata_df[
        "timestamp_stop"
    ].dt.tz_localize(None)

    metadata_df = metadata_df.sort_values("timestamp_start")
    metadata_df["duration"] = (
        metadata_df.timestamp_stop - metadata_df.timestamp_start
    )
    metadata_df["duration_seconds"] = metadata_df[
        "duration"
    ].dt.total_seconds()

    metadata_df["task_simple"] = metadata_df["task"].apply(
        lambda x: x.split(".")[-1]
    )

    parent_dict = (
        metadata_df[["job_id", "task_simple"]]
        .dropna(subset=["job_id"])
        .set_index("job_id")
        .to_dict(orient="index")
    )
    parent_dict.update(
        dict((k, "".join(v.values())) for k, v in parent_dict.items())
    )
    metadata_df["parent_name"] = metadata_df.parent_id.map(parent_dict).fillna(
        ""
    )

    root_idx = metadata_df.parent_id.isnull()
    work_idx = ~metadata_df.job_id.isin(metadata_df.parent_id) | (
        metadata_df.cached & metadata_df.job_id.isnull()
    )

    rename = {"timestamp_start": "timestamp", "timestamp_stop": "timestamp"}
    cols = [
        "cpu",
        "memory",
        "disk_size",
        "preemptible",
        "disk_type",
        "cached",
        "task",
        "task_simple",
    ]

    df1 = pd.concat(
        [
            metadata_df[["timestamp_start"] + cols].rename(columns=rename),
            work_idx.rename("count").astype(int),
        ],
        axis=1,
    )

    df2 = pd.concat(
        [
            metadata_df[["timestamp_stop"] + cols].rename(columns=rename),
            work_idx.rename("count").astype(int) * -1,
        ],
        axis=1,
    )
    df2[["cpu", "memory", "disk_size"]] = (
        df2[["cpu", "memory", "disk_size"]] * -1
    )

    time_df = pd.concat([df1, df2]).sort_values("timestamp")
    time_df["cpu_sum"] = time_df["cpu"].cumsum()
    time_df["memory_sum"] = time_df["memory"].cumsum()
    time_df["disk_size_sum"] = time_df["disk_size"].cumsum()
    time_df["count_sum"] = time_df["count"].cumsum()

    sum_cols = ["cpu_sum", "memory_sum", "disk_size_sum", "count_sum"]
    time_df[sum_cols] = time_df[sum_cols].fillna(method="ffill")

    return metadata_df, time_df, root_idx, work_idx


@st.cache
def get_call_metadata(metadata):
    return get_calls(metadata)


def get_calls(m, alias=None, parent_id=None):
    """
    Modified from download_monitoring_logs.py script by Mark Walker
    https://github.com/broadinstitute/gatk-sv/blob/master/scripts/cromwell/download_monitoring_logs.py
    """

    if isinstance(m, list):
        call_metadata = []
        for m_shard in m:
            call_metadata.extend(
                get_calls(m_shard, alias=alias, parent_id=parent_id)
            )
        return call_metadata

    if "labels" in m:
        alias = add_label_to_alias(alias, m["labels"])

    call_metadata = []
    if ("calls" in m) and m["calls"]:
        if alias:
            name = alias
        else:
            name = m["workflowName"]

        cached = used_cached_results(m)
        start, end = calculate_start_end(m, alias, cached)
        call_metadata.append(
            (
                name,
                start,
                end,
                np.NaN,
                np.NaN,
                None,
                np.NaN,
                None,
                None,
                m["id"],
                parent_id,
            )
        )

        for call in m["calls"]:
            # Skips scatters that don't contain calls
            if "." not in call:
                continue

            # Recursively get metadata
            call_alias = get_call_alias(alias, call)
            call_metadata.extend(
                get_calls(
                    m["calls"][call],
                    alias=call_alias,
                    parent_id=m["id"],
                )
            )

    if "subWorkflowMetadata" in m:
        call_metadata.extend(
            get_calls(
                m["subWorkflowMetadata"],
                alias=alias,
                parent_id=parent_id,
            )
        )

    # In a call
    if alias and ("stderr" in m):
        cached = used_cached_results(m)
        start, end = calculate_start_end(m, alias, cached)
        cpu, memory = get_mem_cpu(m)
        preemptible = was_preemptible_vm(m, cached)
        disk_type, disk_size = get_disk_info(m)

        job_id = None
        if "jobId" in m:
            job_id = m["jobId"]

        call_metadata.append(
            (
                alias,
                start,
                end,
                cpu,
                memory,
                preemptible,
                disk_size,
                disk_type,
                cached,
                job_id,
                parent_id,
            )
        )

    return call_metadata


def calculate_start_end(call_info, alias=None, cached=False):
    """
    Modified from: https://github.com/broadinstitute/dsde-pipelines/blob/develop/scripts/calculate_cost.py
    """
    if "jobId" in call_info:
        job_id = call_info["jobId"].split("/")[-1]
        if alias is None or alias == "":
            alias = job_id
        else:
            alias += "." + job_id
    elif alias is None or alias == "":
        alias = "NA"

    # get start (start time of VM start) & end time (end time of 'ok') according to metadata
    start = pd.NaT
    end = pd.NaT

    if "executionEvents" in call_info:
        for x in call_info["executionEvents"]:
            # ignore incomplete executionEvents (could be due to server restart or similar)
            if "description" not in x:
                continue
            y = x["description"]

            if y.startswith("PreparingJob"):
                start = dateutil.parser.parse(x["startTime"])

            if "backend" in call_info and call_info["backend"] == "PAPIv2":
                if y.startswith("Worker released") and not cached:
                    end = dateutil.parser.parse(x["endTime"])
                elif y.startswith("CallCacheReading") and cached:
                    end = dateutil.parser.parse(x["endTime"])
            else:
                if y.startswith("UpdatingJobStore") and not cached:
                    end = dateutil.parser.parse(x["endTime"])
                elif y.startswith("CallCacheReading") and cached:
                    end = dateutil.parser.parse(x["endTime"])

    # if we are preempted or if cromwell used previously cached results, we don't even get a start time from JES.
    # if cromwell was restarted, the start time from JES might not have been written to the metadata.
    # in either case, use the Cromwell start time which is earlier but not wrong.

    if pd.isnull(start):
        start = dateutil.parser.parse(call_info["start"])

    # if we are preempted or if cromwell used previously cached results, we don't get an endTime from JES right now.
    # if cromwell was restarted, the start time from JES might not have been written to the metadata.
    # in either case, use the Cromwell end time which is later but not wrong
    if pd.isnull(end):
        if "end" in call_info:
            end = dateutil.parser.parse(call_info["end"])
        # elif override_warning:
        #     logging.warning(
        #         "End time not found, omitting job {}".format(alias)
        #     )
        #     end = start
        # else:
        #     raise RuntimeError(
        #         (
        #             f"End time not found for job {alias} (may be running or have been aborted)."
        #             " Run again with --override-warning to continue anyway and omit the job."
        #         )
        #     )

    return start, end


def get_mem_cpu(m):
    """
    Modified from: https://github.com/broadinstitute/dsde-pipelines/blob/develop/scripts/calculate_cost.py
    """
    cpu = np.NaN
    memory = np.NaN
    if "runtimeAttributes" in m:
        if "cpu" in m["runtimeAttributes"]:
            cpu = int(m["runtimeAttributes"]["cpu"])
        if "memory" in m["runtimeAttributes"]:
            mem_str = m["runtimeAttributes"]["memory"]
            memory = float(mem_str.split()[0])
    return cpu, memory


def used_cached_results(metadata):
    """
    Modified from: https://github.com/broadinstitute/dsde-pipelines/blob/develop/scripts/calculate_cost.py
    """
    return (
        "callCaching" in metadata
        and "hit" in metadata["callCaching"]
        and metadata["callCaching"]["hit"]
    )


def get_disk_info(metadata):
    """
    Modified from: https://github.com/broadinstitute/dsde-pipelines/blob/develop/scripts/calculate_cost.py
    """

    boot_disk_size = 0.0
    disk_size = 0.0
    disk_type = "HDD"

    # GCP and AWS backends handle disk information in a different way
    try:
        if (
            "runtimeAttributes" in metadata
            and "disks" in metadata["runtimeAttributes"]
        ):  # GCP
            if "bootDiskSizeGb" in metadata["runtimeAttributes"]:
                boot_disk_size = metadata["runtimeAttributes"][
                    "bootDiskSizeGb"
                ]

            (_, disk_size, disk_type) = metadata["runtimeAttributes"][
                "disks"
            ].split()
    except ValueError:
        if "inputs" in metadata:  # AWS
            if (
                "runtime_attr" in metadata["inputs"]
                and "disk_gb" in metadata["inputs"]["runtime_attr"]
            ):
                if "boot_disk_gb" in metadata["inputs"]["runtime_attr"]:
                    boot_disk_size = metadata["inputs"]["runtime_attr"][
                        "boot_disk_gb"
                    ]

                disk_size = metadata["inputs"]["runtime_attr"]["disk_gb"]
            elif "disk_size" in metadata["inputs"]:
                disk_size = metadata["inputs"]["disk_size"]

    return disk_type, float(boot_disk_size) + float(disk_size)


def was_preemptible_vm(metadata, was_cached):
    """
    Modified from: https://github.com/broadinstitute/dsde-pipelines/blob/develop/scripts/calculate_cost.py
    """
    # if call cached, not any type of VM, but don't inflate nonpreemptible count
    if was_cached:
        return None
    elif (
        "runtimeAttributes" in metadata
        and "preemptible" in metadata["runtimeAttributes"]
    ):
        pe_count = int(metadata["runtimeAttributes"]["preemptible"])
        attempt = int(metadata["attempt"])

        return attempt <= pe_count
    else:
        # we can't tell (older metadata) so conservatively return false
        return False


def validate_workflow(metadata):
    if "status" not in metadata:
        message = (
            "Incomplete metadata input file. File lacks workflow status field."
        )
        return STATUS.INCOMPLETE, message

    # Unrecognized workflow ID failure - unable to download metadata
    if metadata["status"] == "fail":
        message = "Workflow metadata download failure."
        if "message" in metadata:
            message += f" Message: {metadata['message']}"
        return STATUS.UNRECOGNIZED, message

    if metadata["status"] == "Failed":
        message = (
            "Workflow failed, which is likely to provide inaccurate results. "
        )
        if "failures" in metadata:
            fail_list = []
            for failures in metadata["failures"]:
                if "message" in failures:
                    fail_list.append(failures["message"])
            if fail_list:
                message += "Messages: " + "; ".join(fail_list)
        return STATUS.FAILED, message

    if metadata["status"] == "Aborted":
        message = (
            "Workflow aborted, which is likely to provide inaccurate results."
        )
        return STATUS.ABORTED, message

    if metadata["status"] == "Running":
        message = "Workflow is still running, which is likely to provide inaccurate results."
        return STATUS.RUNNING, message

    if metadata["status"] == "Succeeded":
        return STATUS.SUCCEEDED, None


def add_label_to_alias(alias, labels):
    # In alias, track hierarchy of workflow/task up to current task nicely without repetition
    if alias is None:
        alias = ""
    to_add = ""
    if "wdl-call-alias" in labels:
        to_add = labels["wdl-call-alias"]
    elif "wdl-task-name" in labels:
        to_add = labels["wdl-task-name"]
    if to_add != "" and not alias.endswith(to_add):
        if alias != "" and alias[-1] != ".":
            alias += "."
        alias += to_add

    return alias


def get_call_alias(alias, call):
    # In call_alias, track hierarchy of workflow/task up to current call nicely without repetition
    if alias is None:
        alias = ""
    call_split = call.split(".")
    call_name = call
    if alias.endswith(call_split[0]):
        call_name = call_split[1]
    call_alias = alias
    if call_alias != "" and call_alias[-1] != ".":
        call_alias += "."
    call_alias += call_name

    return call_alias


if __name__ == "__main__":
    main()
