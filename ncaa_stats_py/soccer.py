# Author: Derek Willis (dwillis@gmail.com)
# File Name: `soccer.py`
# Purpose: Houses functions that allows one to access NCAA soccer data
# Creation Date: 2024-09-20 08:15 PM EDT


import logging
import re
from datetime import date, datetime
from os import mkdir
from os.path import exists, expanduser, getmtime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from ncaa_stats_py.utls import (
    _format_folder_str,
    _get_schools,
    _get_seconds_from_time_str,
    _get_stat_id,
    _get_webpage,
)

def get_soccer_teams(
	season: int, level: str | int,
	get_womens_soccer_data: bool = False
) -> pd.DataFrame:
	"""
	Retrieves a list of soccer teams from the NCAA.

	Parameters
	----------
	`season` (int, mandatory): season year
	`level` (int or str, mandatory): division (1/2/3 or "I"/"II"/"III")
	`get_womens_soccer_data` (bool, optional): toggles women's data

	Notes
	-----
	This function mirrors the style of `lacrosse.get_lacrosse_teams`.
	Uses `WSO` and `MSO` for team sport codes per repository convention.
	The cache directory is `.ncaa_stats_py/soccer_{sport_id}`.
	"""
	sport_id = ""
	load_from_cache = True
	home_dir = expanduser("~")
	home_dir = _format_folder_str(home_dir)
	teams_df = pd.DataFrame()
	teams_df_arr = []
	temp_df = pd.DataFrame()
	formatted_level = ""
	ncaa_level = 0

	# Use scoring offense as stat_sequence
	if get_womens_soccer_data is True:
		sport_code = "WSO"
		stat_sequence = 56
	else:
		# mens mapping in utls does not contain a 'team' stat id; keep string
		sport_code = "MSO"
		stat_sequence = 30

	# Normalize level input
	if isinstance(level, int) and level == 1:
		formatted_level = "I"
		ncaa_level = 1
	elif isinstance(level, int) and level == 2:
		formatted_level = "II"
		ncaa_level = 2
	elif isinstance(level, int) and level == 3:
		formatted_level = "III"
		ncaa_level = 3
	elif isinstance(level, str) and (
		level.lower() in {"i", "d1", "1"}
	):
		formatted_level = "I"
		ncaa_level = 1
	elif isinstance(level, str) and (
		level.lower() in {"ii", "d2", "2"}
	):
		formatted_level = "II"
		ncaa_level = 2
	elif isinstance(level, str) and (
		level.lower() in {"iii", "d3", "3"}
	):
		formatted_level = "III"
		ncaa_level = 3
	else:
		raise ValueError("Invalid 'level' parameter for get_soccer_teams")

	# Ensure cache directories exist
	if exists(f"{home_dir}/.ncaa_stats_py/"):
		pass
	else:
		mkdir(f"{home_dir}/.ncaa_stats_py/")

	if exists(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/"):
		pass
	else:
		mkdir(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/")

	if exists(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/teams/"):
		pass
	else:
		mkdir(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/teams/")

	cache_file = (
		f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/teams/"
		+ f"{season}_{formatted_level}_teams.csv"
	)

	if exists(cache_file):
		teams_df = pd.read_csv(cache_file)
		file_mod_datetime = datetime.fromtimestamp(getmtime(cache_file))
	else:
		file_mod_datetime = datetime.today()
		load_from_cache = False

	now = datetime.today()
	age = now - file_mod_datetime

	if (
		age.days > 1 and
		season >= now.year and
		now.month <= 7
	):
		load_from_cache = False
	elif (
		age.days >= 14 and
		season >= (now.year - 1) and
		now.month <= 7
	):
		load_from_cache = False
	elif age.days >= 35:
		load_from_cache = False

	if load_from_cache is True:
		return teams_df

	logging.warning(
		f"Either we could not load {season} D{level} schools from cache, "
		+ "or it's time to refresh the cached data."
	)

	schools_df = _get_schools()
	change_url = (
		"https://stats.ncaa.org/rankings/change_sport_year_div?"
		+ f"academic_year={season}&division={ncaa_level}.0"
		+ f"&sport_code={sport_code}"
	)

	inst_url = (
		"https://stats.ncaa.org/rankings/institution_trends?"
		+ f"academic_year={season}&division={ncaa_level}.0&"
		+ f"ranking_period=0&sport_code={sport_code}"
		+ (f"&stat_seq={stat_sequence}" if stat_sequence else "")
	)

	# Use the shared _get_webpage utility for all requests (like lacrosse.py)
	inst_html = _get_webpage(inst_url)
	if not inst_html:
		logging.info("Failed to fetch institution_trends page")
		return pd.DataFrame()
	soup = BeautifulSoup(inst_html, features="lxml")

	# Prefer Playwright DOM-extraction for structured rows when possible
	# (Removed _playwright_extract_teams; use soup-based parsing only)

	# Try parsing common table layouts used on stats.ncaa.org
	try:
		# Prefer extracting team rows by finding anchors that link to /teams/<id>
		# This is more robust than relying on column positions since layouts vary.
		anchors = soup.find_all('a', href=True)
		found = {}
		for a in anchors:
			href = a.get('href')
			if not href or '/teams/' not in href:
				continue
			# Heuristics to ensure this anchor is a team link shown in rankings
			is_team_anchor = False
			# target="TEAM_WIN" is used on the rankings table
			if a.get('target') and a.get('target').upper() == 'TEAM_WIN':
				is_team_anchor = True
			# class 'skipMask' appears on team anchors
			if not is_team_anchor:
				cls = a.get('class') or []
				if isinstance(cls, list) and 'skipMask' in cls:
					is_team_anchor = True
			# extract team id
			try:
				team_id = int(href.split('/teams/')[1].split('/')[0])
			except Exception:
				team_id = None
			# find row and parent table
			tr = a.find_parent('tr')
			if tr is None:
				continue
			table = tr.find_parent('table')
			# check parent td classes (e.g., 'sorting_1')
			if not is_team_anchor:
				parent_td = a.find_parent('td')
				if parent_td is not None:
					pcls = parent_td.get('class') or []
					if isinstance(pcls, list) and any('sorting' in c for c in pcls):
						is_team_anchor = True
			if not is_team_anchor:
				continue
			# ensure this table looks like a rankings table (has Team/School header or known id)
			is_rankings = False
			if table is not None:
				tid = table.get('id')
				if tid in ('stat_grid', 'rankings_table'):
					is_rankings = True
				else:
					thead = table.find('thead')
					if thead:
						head_txt = ' '.join([th.get_text(' ', strip=True).lower() for th in thead.find_all('th')])
						if any(k in head_txt for k in ('team', 'school', 'institution')):
							is_rankings = True
			# skip anchors that are not in a likely rankings table
			if not is_rankings:
				continue
			# extract name and conference heuristics
			team_name = a.get_text(strip=True)
			team_conf = ''
			tds = tr.find_all('td')
			if tds:
				# prefer data-order attr if present
				for cell in tds:
					data_order = cell.get('data-order')
					if data_order and ',' in data_order:
						parts = [p.strip() for p in data_order.split(',', 1)]
						if len(parts) == 2:
							team_name = parts[0]
							team_conf = parts[1]
							break
				# fallback conference in next td if available
				if not team_conf and len(tds) > 1:
					team_conf = tds[1].get_text(strip=True)
			# normalize and dedupe
			if team_name:
				key = (team_id, team_name)
				if key in found:
					continue
				found[key] = True
				temp_df = pd.DataFrame({
					"season": season,
					"ncaa_division": ncaa_level,
					"ncaa_division_formatted": formatted_level,
					"team_conference_name": team_conf,
					"team_id": team_id,
					"school_name": team_name,
					"sport_id": sport_id,
				}, index=[0])
				teams_df_arr.append(temp_df)

		# If anchor-based extraction found rows, skip the older heuristics
		if teams_df_arr:
			pass
		# fallback: previous table-based heuristics (keep as-is)
			table = soup.find("table", {"id": "stat_grid"})
			if table is not None:
				tbody = table.find("tbody")
				t_rows = tbody.find_all("tr") if tbody is not None else []
				for t in t_rows:
					# Attempt several heuristics to extract team id, name, conference
					team_id = None
					team_name = None
					team_conf = ""
					# 1) look for anchor with /teams/ID
					try:
						a = t.find('a', href=True)
						if a and '/teams/' in a['href']:
							try:
								team_id = int(a['href'].split('/teams/')[1].split('/')[0])
							except Exception:
								team_id = None
							team_name = a.get_text(strip=True) or None
					except Exception:
						pass
					# 2) some tables embed team,conference in data-order attr on a td
					if not team_name:
						try:
							td = t.find_all('td')
							for cell in td:
								data_order = cell.get('data-order')
								if data_order and ',' in data_order:
									team_name, team_conf = [x.strip() for x in data_order.split(',', 1)]
									break
							# fallback: visible text in first or second td
							if not team_name and td:
								if len(td) > 1:
									team_name = td[1].get_text(strip=True)
								else:
									team_name = td[0].get_text(strip=True)
						except Exception:
							pass
					# 3) if team_name is numeric-like (a stat), skip this row
					if team_name:
						try:
							float(team_name.replace(',', ''))
							# numeric-only; skip
							continue
						except Exception:
							# not numeric, ok
							pass
					# final guard: must have a team_name
					if not team_name:
						continue
					# normalize
					team_name = team_name.strip()
					temp_df = pd.DataFrame(
						{
							"season": season,
							"ncaa_division": ncaa_level,
							"ncaa_division_formatted": formatted_level,
							"team_conference_name": team_conf,
							"team_id": team_id,
							"school_name": team_name,
							"sport_id": sport_id,
						},
						index=[0],
					)
					teams_df_arr.append(temp_df)
		else:
			# Try the rankings_table layout
			table = soup.find("table", {"id": "rankings_table"})
			if table is not None:
				tbody = table.find("tbody")
				t_rows = tbody.find_all("tr") if tbody is not None else []
				for t in t_rows:
					try:
						team_link = t.find("a")
						team_id = int(team_link.get("href").replace("/teams/", ""))
					except Exception:
						continue
					# some rows embed team,conference in data-order attribute
					team = t.find_all("td")[1].get("data-order")
					if team and "," in team:
						team_name, team_conf = team.split(",", 1)
					else:
						# fallback: visible text
						team_name = t.find_all("td")[1].text.strip()
						team_conf = ""
					temp_df = pd.DataFrame(
						{
							"season": season,
							"ncaa_division": ncaa_level,
							"ncaa_division_formatted": formatted_level,
							"team_conference_name": team_conf,
							"team_id": team_id,
							"school_name": team_name,
							"sport_id": sport_id,
						},
						index=[0],
					)
					teams_df_arr.append(temp_df)
			else:
				# Generic table parse: first <table> on page
				table = soup.find("table")
				if table is not None:
					rows = table.find_all("tr")
					for r in rows:
						cols = r.find_all("td")
						if not cols:
							continue
						# try extract link and name
						try:
							team_link = r.find("a")
							team_id = int(team_link.get("href").replace("/teams/", ""))
						except Exception:
							team_id = None
						school_name = cols[1].get_text(strip=True) if len(cols) > 1 else cols[0].get_text(strip=True)
						temp_df = pd.DataFrame(
							{
								"season": season,
								"ncaa_division": ncaa_level,
								"ncaa_division_formatted": formatted_level,
								"team_conference_name": "",
								"team_id": team_id,
								"school_name": school_name,
								"sport_id": sport_id,
							},
							index=[0],
						)
						teams_df_arr.append(temp_df)
	except Exception as e:
		logging.warning(f"Failed to parse soccer teams: {e}")

	if not teams_df_arr:
		return pd.DataFrame()

	teams_df = pd.concat(teams_df_arr, ignore_index=True)
	teams_df = pd.merge(
		left=teams_df,
		right=schools_df,
		on=["school_name"],
		how="left",
	)
	teams_df.sort_values(by=["team_id"], inplace=True)

	teams_df.to_csv(cache_file, index=False)

	return teams_df


def load_soccer_teams(
	start_year: int = 2010,
	get_womens_soccer_data: bool = False
) -> pd.DataFrame:
	"""
	Compiles a list of known NCAA soccer teams from `start_year` to present.
	"""
	if get_womens_soccer_data is True:
		sport_id = "MSO"
	else:
		sport_id = "WSO"

	teams_df_arr = []
	now = datetime.now()
	ncaa_seasons = [x for x in range(start_year, (now.year + 1))]

	logging.info("Loading soccer teams across seasons; this may take a while")
	for s in tqdm(ncaa_seasons, desc="seasons"):
		for div in ["I", "II", "III"]:
			try:
				df = get_soccer_teams(s, div, get_womens_soccer_data=get_womens_soccer_data)
				if df is not None and not df.empty:
					df["season"] = s
					df["ncaa_division"] = div
					teams_df_arr.append(df)
			except Exception:
				logging.debug(f"Skipping season {s} division {div} due to error")

	if teams_df_arr:
		teams_df = pd.concat(teams_df_arr, ignore_index=True)
		teams_df = teams_df.infer_objects()
		return teams_df

	return pd.DataFrame()


def _playwright_fetch(url: str, wait_for: str | None = None) -> str | None:
	"""Render a URL with Playwright and return page.content() or None on failure.
	"""
	try:
		from playwright.sync_api import sync_playwright
		with sync_playwright() as pw:
			browser = pw.chromium.launch(headless=True)
			context = browser.new_context()
			page = context.new_page()
			page.goto(url, timeout=45000)
			if wait_for:
				try:
					page.wait_for_selector(wait_for, timeout=30000)
				except Exception:
					pass
			content = page.content()
			try:
				browser.close()
			except Exception:
				pass
			return content
	except Exception:
		return None


def get_soccer_team_schedule(team_id: int) -> pd.DataFrame:
	"""
	Retrieves a team schedule, from a valid NCAA soccer team ID.
	"""
	sport_id = "WSO"
	games_df = pd.DataFrame()
	games_df_arr = []
	schools_df = _get_schools()
	home_dir = expanduser("~")
	home_dir = _format_folder_str(home_dir)

	# Determine sport_id via cached schools if possible (lightweight)
	url = f"https://stats.ncaa.org/teams/{team_id}"
	try:
		response_html = _playwright_fetch(url, wait_for='table')
		if not response_html:
			logging.warning("Failed to render team page with Playwright")
			return games_df
	except Exception:
		logging.warning("Failed to fetch team page")
		return games_df

	soup = BeautifulSoup(response_html, features="lxml")
	# Minimal parse: find schedule table and walk rows
	try:
		table = soup.find("table")
		if table:
			rows = table.find_all("tr")
			for r in rows:
				cols = [c.get_text(strip=True) for c in r.find_all("td")]
				if not cols:
					continue
				games_df_arr.append({
					"game_date": cols[0] if len(cols) > 0 else None,
					"opponent": cols[1] if len(cols) > 1 else None,
					"result": cols[2] if len(cols) > 2 else None,
				})
	except Exception:
		logging.warning("Failed to parse team schedule")

	if games_df_arr:
		games_df = pd.DataFrame(games_df_arr)
		# join school ids
		temp_df = schools_df.rename(columns={"school_name": "opponent", "school_id": "opponent_school_id"})
		games_df = games_df.merge(right=temp_df, on="opponent", how="left")
		games_df.to_csv(
			f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/team_schedule/{team_id}_team_schedule.csv",
			index=False,
		)

	return games_df


def get_soccer_day_schedule(
	game_date: str | date | datetime,
	level: str | int = "I",
	get_womens_soccer_data: bool = False
):
	# Placeholder: implement similar to lacrosse.get_lacrosse_day_schedule
	pass


def get_full_soccer_schedule(
	season: int,
	level: str | int = "I",
	get_womens_soccer_data: bool = False
) -> pd.DataFrame:
	# Placeholder: implement full season aggregation
	pass


def get_soccer_team_roster(team_id: int) -> pd.DataFrame:
	pass


def get_soccer_player_season_stats(
	team_id: int,
	season: int,
	level: str | int,
	get_womens_soccer_data: bool = False
) -> pd.DataFrame:
	pass


def get_soccer_player_game_stats(player_id: int) -> pd.DataFrame:
	pass


def get_soccer_game_player_stats(game_id: int) -> pd.DataFrame:
	pass


def get_soccer_raw_pbp(game_id: int) -> pd.DataFrame:
	pass


def get_soccer_team_stats(
	season: int,
	level: str | int = "I",
	get_womens_soccer_data: bool = False
) -> pd.DataFrame:
	pass


if __name__ == "__main__":
	pass
