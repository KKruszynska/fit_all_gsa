# fit_all_gsa
To use:
1. Git clone this repository.
2. Move to fit_all_gsa folder.
3. Install required packages using `pip install -r requirements.txt`
4. You may want to git clone and install the most recent version of pyLIMA directly from the pyLIMA repo: https://github.com/ebachelet/pyLIMA

Run the program as follows:

`python fit_all_gsa_py3.py start_time_in_yyyy-mm-dd end_time_in_yyyy-mm-dd output_file_name_base`

Or, to turn off high-cadence zone exclusion:

`python fit_all_gsa_py3.py tart_time_in_yyyy-mm-dd end_time_in_yyyy-mm-dd output_file_name_base hcz_zone=False`

Let me know if sth breaks down.
