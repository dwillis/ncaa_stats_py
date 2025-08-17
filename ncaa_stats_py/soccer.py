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
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    teams_df = pd.DataFrame()
    teams_df_arr = []
    temp_df = pd.DataFrame()
    formatted_level = ""
    ncaa_level = 0

    # Set sport codes and keys for stat lookup
    if get_womens_soccer_data is True:
        sport_code = "WSO"
        sport_key = "womens_soccer"
    else:
        sport_code = "MSO"
        sport_key = "mens_soccer"

    # For soccer: user season (2025) means Fall 2025 = Academic Year 2025-26
    stat_dict_season = season

    try:
        stat_sequence = _get_stat_id(sport_key, stat_dict_season, "team")
    except LookupError:
        logging.warning(f"Could not find team stat ID for {sport_key} season {stat_dict_season}")
        # Fallback to hardcoded values
        if get_womens_soccer_data:
            stat_sequence = 56
        else:
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

    # Ensure cache directories exist (using os.makedirs for robustness)
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    teams_cache_dir = f"{soccer_cache_dir}/teams"
    for d in [base_cache_dir, soccer_cache_dir, teams_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    cache_file = f"{teams_cache_dir}/{season}_{formatted_level}_teams.csv"

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
    # Deduplicate schools_df on school_name before merging
    schools_df = schools_df.drop_duplicates(subset=["school_name"]).copy()
    
    change_url = (
        "https://stats.ncaa.org/rankings/change_sport_year_div?"
        + f"academic_year={season + 1}.0&division={ncaa_level}.0"
        + f"&sport_code={sport_code}"
    )

    inst_url = (
        "https://stats.ncaa.org/rankings/institution_trends?"
        + f"academic_year={season + 1}.0&division={ncaa_level}.0&"
        + f"ranking_period=0&sport_code={sport_code}"
        + f"&stat_seq={stat_sequence}"
    )

    # Use the shared _get_webpage utility for all requests (like lacrosse.py)
    inst_html = _get_webpage(inst_url)
    soup = BeautifulSoup(inst_html.text, features="lxml")

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
        if not teams_df_arr:
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
    # Deduplicate merged teams_df on season and team_id
    teams_df = teams_df.drop_duplicates(subset=["season", "team_id"]).copy()
    teams_df.sort_values(by=["team_id"], inplace=True)

    # Ensure the teams cache directory exists before writing
    teams_cache_dir = os.path.dirname(cache_file)
    if not os.path.exists(teams_cache_dir):
        os.makedirs(teams_cache_dir)
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
        sport_id = "WSO"
    else:
        sport_id = "MSO"

    teams_df_arr = []
    now = datetime.now()
    ncaa_seasons = [x for x in range(start_year, (now.year))]

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

def get_soccer_team_schedule(team_id: int) -> pd.DataFrame:
    """
    Retrieves a team schedule, from a valid NCAA soccer team ID.
    """

    sport_id = ""
    schools_df = _get_schools()
    games_df = pd.DataFrame()
    games_df_arr = []
    season = 0
    temp_df = pd.DataFrame()
    load_from_cache = True

    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    url = f"https://stats.ncaa.org/teams/{team_id}"

    # Try to get team info from cached data, with proper error handling
    team_info_found = False
    
    try:
        team_df = load_soccer_teams()
        team_df = team_df[team_df["team_id"] == team_id]
        if not team_df.empty:
            season = team_df["season"].iloc[0]
            ncaa_division = team_df["ncaa_division"].iloc[0]
            ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
            sport_id = "MSO"
            team_info_found = True
        del team_df
    except Exception as e:
        logging.info(f"Could not find team in men's soccer data: {e}")

    if not team_info_found:
        try:
            team_df = load_soccer_teams(get_womens_soccer_data=True)
            team_df = team_df[team_df["team_id"] == team_id]
            if not team_df.empty:
                season = team_df["season"].iloc[0]
                ncaa_division = team_df["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
                sport_id = "WSO"
                team_info_found = True
            del team_df
        except Exception as e:
            logging.info(f"Could not find team in women's soccer data: {e}")

    # If team not found in cached data, extract info from the team page directly
    if not team_info_found:
        logging.warning(f"Team ID {team_id} not found in cached teams data. Extracting info from team page.")
        try:
            # Simplified - just call _get_webpage directly
            response = _get_webpage(url=url)
            soup = BeautifulSoup(response.text, features="lxml")
            
            # Extract season from the year selector
            season_element = soup.find("select", {"id": "year_list"})
            if season_element:
                selected_option = season_element.find("option", {"selected": "selected"})
                if selected_option:
                    season_name = selected_option.text
                    # For soccer: extract the fall year from academic year format
                    # "2023-24" means Fall 2023 season, so we want 2023
                    if "-" in season_name:
                        season = int(season_name.split("-")[0])
                    else:
                        season = int(season_name)
                else:
                    season = datetime.today().year
            else:
                season = datetime.today().year
            
            # Try to determine sport from page content
            page_text = soup.get_text().lower()
            if "women" in page_text or "wso" in page_text:
                sport_id = "WSO"
            else:
                sport_id = "MSO"  # Default to men's
            
            # Default division info
            ncaa_division = 1
            ncaa_division_formatted = "I"
            
            logging.info(f"Extracted info for team {team_id}: season={season}, sport_id={sport_id}")
            
        except Exception as e:
            logging.error(f"Could not extract team info from page: {e}")
            # Ultimate fallback
            season = datetime.today().year
            sport_id = "MSO"
            ncaa_division = 1
            ncaa_division_formatted = "I"

    # Now that we have season and sport info, try to get the proper stat ID
    stat_sequence = None
    if season and sport_id:
        try:
            sport_key = "womens_soccer" if sport_id == "WSO" else "mens_soccer"
            stat_sequence = _get_stat_id(sport_key, season, "team")
            logging.info(f"Found stat ID {stat_sequence} for {sport_key} season {season}")
        except LookupError:
            logging.warning(f"Could not find team stat ID for {sport_key} season {season}")
            # Use fallback values
            stat_sequence = 56 if sport_id == "WSO" else 30

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/")

    if exists(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/team_schedule/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/team_schedule/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/team_schedule/"
        + f"{team_id}_team_schedule.csv"
    ):
        games_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/soccer_{sport_id}/team_schedule/"
            + f"{team_id}_team_schedule.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/"
                + f"soccer_{sport_id}/team_schedule/"
                + f"{team_id}_team_schedule.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime
    if age.days > 1 and season >= now.year and now.month <= 7:
        load_from_cache = False

    if load_from_cache is True:
        return games_df

    # Simplified - just call _get_webpage directly
    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
    season_name = (
        soup.find("select", {"id": "year_list"})
        .find("option", {"selected": "selected"})
        .text
    )
    # For NCAA soccer, the season always starts and ends in the fall semester
    # Thus, if `season_name` = "2023-24", this is the "2023" soccer season,
    # because soccer runs entirely in the fall semester
    soup = soup.find_all(
        "div",
        {"class": "col p-0"},
    )

    # declaring it here to prevent potential problems down the road.
    table_data = ""
    for s in soup:
        try:
            temp_name = s.find("div", {"class": "card-header"})
            temp_name = temp_name.text
        except Exception as e:
            logging.warning(
                f"Could not parse card header. Full exception `{e}`. "
                + "Attempting alternate method."
            )
            temp_name = s.find("tr", {"class": "heading"}).find("td").text

        if "schedule" in temp_name.lower():
            table_data = s.find("table")

    t_rows = table_data.find_all("tr", {"class": "underline_rows"})

    if len(t_rows) == 0:
        t_rows = table_data.find_all("tr")

    for g in t_rows:
        is_valid_row = True
        game_num = 1
        ot_periods = 0
        is_home_game = True
        is_neutral_game = False

        cells = g.find_all("td")
        if len(cells) <= 1:
            continue

        game_date = cells[0].text

        # If "(" is in the same cell as the date,
        # this means that this game is part of a series.
        # The number encased in `()` is the game number in the series.
        if "(" in game_date:
            game_date = game_date.replace(")", "")
            game_date, game_num = game_date.split("(")
            game_date = game_date.strip()
            game_num = int(game_num.strip())

        game_date = datetime.strptime(game_date, "%m/%d/%Y").date()

        try:
            opp_team_id = cells[1].find("a").get("href")
        except IndexError:
            logging.info("Skipping row because it is clearly not a row that has schedule data.")
            is_valid_row = False
        except AttributeError as e:
            logging.info(f"Could not extract a team ID for this game. Full exception {e}")
            opp_team_id = "-1"
        except Exception as e:
            logging.warning(f"An unhandled exception has occurred when trying to get the opposition team ID for this game. Full exception `{e}`.")
            raise e
            
        if is_valid_row is True:
            if opp_team_id is not None:
                opp_team_id = opp_team_id.replace("/teams/", "")
                opp_team_id = int(opp_team_id)

                try:
                    opp_team_name = cells[1].find("img").get("alt")
                except AttributeError:
                    logging.info("Couldn't find the opposition team name for this row from an image element. Attempting a backup method")
                    opp_team_name = cells[1].text
                except Exception as e:
                    logging.info(f"Unhandled exception when trying to get the opposition team name from this game. Full exception `{e}`")
                    raise e
            else:
                opp_team_name = cells[1].text

            if opp_team_name and opp_team_name[0] == "@":
                opp_team_name = opp_team_name.strip().replace("@", "")
            elif opp_team_name and "@" in opp_team_name:
                opp_team_name = opp_team_name.strip().split("@")[0]

            opp_text = cells[1].text
            opp_text = opp_text.strip()
            if "@" in opp_text and opp_text[0] == "@":
                is_home_game = False
            elif "@" in opp_text and opp_text[0] != "@":
                is_neutral_game = True
                is_home_game = False
            elif "championship" in opp_text.lower():
                is_neutral_game = True
                is_home_game = False
            elif "ncaa" in opp_text.lower():
                is_neutral_game = True
                is_home_game = False

            del opp_text

            score = cells[2].text.strip()
            if len(score) == 0:
                score_1 = 0
                score_2 = 0
            elif (
                "canceled" not in score.lower() and
                "ppd" not in score.lower()
            ):
                # Handle overtime notation in soccer scores
                score = score.replace("(-3 OT)", "")
                score_1, score_2 = score.split("-")

                # Remove "W", "L", or "T" from score_1
                if any(x in score_1 for x in ["W", "L", "T"]):
                    score_1 = score_1.split(" ")[1]

                if "(" in score_2:
                    score_2 = score_2.replace(")", "")
                    score_2, ot_periods = score_2.split("(")
                    ot_periods = ot_periods.replace("OT", "")
                    ot_periods = ot_periods.replace(" ", "")
                    ot_periods = int(ot_periods)

                if ot_periods is None:
                    ot_periods = 0
                score_1 = int(score_1)
                score_2 = int(score_2)
            else:
                score_1 = None
                score_2 = None

            try:
                game_id = cells[2].find("a").get("href")
                game_id = game_id.replace("/contests", "")
                game_id = game_id.replace("/box_score", "")
                game_id = game_id.replace("/", "")
                game_id = int(game_id)
                game_url = f"https://stats.ncaa.org/contests/{game_id}/box_score"
            except AttributeError as e:
                logging.info(f"Could not parse a game ID for this game. Full exception `{e}`.")
                game_id = None
                game_url = None
            except Exception as e:
                logging.info(f"An unhandled exception occurred when trying to find a game ID for this game. Full exception `{e}`.")
                raise e
                
            try:
                attendance = cells[3].text
                attendance = attendance.replace(",", "")
                attendance = attendance.replace("\n", "")
                attendance = int(attendance)
            except (IndexError, ValueError) as e:
                logging.info(f"Could not parse attendance for this game. Full exception `{e}`.")
                attendance = None
            except Exception as e:
                logging.info(f"An unhandled exception occurred when trying to find this game's attendance. Full exception `{e}`.")
                raise e

            if is_home_game is True:
                temp_df = pd.DataFrame(
                    {
                        "season": season,
                        "season_name": season_name,
                        "game_id": game_id,
                        "game_date": game_date,
                        "game_num": game_num,
                        "ot_periods": ot_periods,
                        "home_team_id": team_id,
                        "home_team_name": school_name,
                        "away_team_id": opp_team_id,
                        "away_team_name": opp_team_name,
                        "home_team_score": score_1,
                        "away_team_score": score_2,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                    },
                    index=[0],
                )
                games_df_arr.append(temp_df)
                del temp_df
            elif is_neutral_game is True:
                # Order team IDs for consistent neutral game representation
                t_ids = [opp_team_id, team_id]
                t_ids.sort()

                if t_ids[0] == team_id:
                    # home
                    temp_df = pd.DataFrame(
                        {
                            "season": season,
                            "season_name": season_name,
                            "game_id": game_id,
                            "game_date": game_date,
                            "game_num": game_num,
                            "ot_periods": ot_periods,
                            "home_team_id": team_id,
                            "home_team_name": school_name,
                            "away_team_id": opp_team_id,
                            "away_team_name": opp_team_name,
                            "home_team_score": score_1,
                            "away_team_score": score_2,
                            "is_neutral_game": is_neutral_game,
                            "game_url": game_url,
                        },
                        index=[0],
                    )
                else:
                    # away
                    temp_df = pd.DataFrame(
                        {
                            "season": season,
                            "season_name": season_name,
                            "game_id": game_id,
                            "game_date": game_date,
                            "game_num": game_num,
                            "ot_periods": ot_periods,
                            "home_team_id": opp_team_id,
                            "home_team_name": opp_team_name,
                            "away_team_id": team_id,
                            "away_team_name": school_name,
                            "home_team_score": score_2,
                            "away_team_score": score_1,
                            "is_neutral_game": is_neutral_game,
                            "game_url": game_url,
                        },
                        index=[0],
                    )

                games_df_arr.append(temp_df)
                del temp_df
            else:
                temp_df = pd.DataFrame(
                    {
                        "season": season,
                        "season_name": season_name,
                        "game_id": game_id,
                        "game_date": game_date,
                        "game_num": game_num,
                        "ot_periods": ot_periods,
                        "home_team_id": opp_team_id,
                        "home_team_name": opp_team_name,
                        "away_team_id": team_id,
                        "away_team_name": school_name,
                        "home_team_score": score_2,
                        "away_team_score": score_1,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                    },
                    index=[0],
                )

                games_df_arr.append(temp_df)
                del temp_df

    games_df = pd.concat(games_df_arr, ignore_index=True)

    temp_df = schools_df.rename(
        columns={
            "school_name": "home_team_name",
            "school_id": "home_school_id"
        }
    )
    games_df = games_df.merge(right=temp_df, on="home_team_name", how="left")

    temp_df = schools_df.rename(
        columns={
            "school_name": "away_team_name",
            "school_id": "away_school_id"
        }
    )
    games_df = games_df.merge(right=temp_df, on="away_team_name", how="left")
    games_df["ncaa_division"] = ncaa_division
    games_df["ncaa_division_formatted"] = ncaa_division_formatted

    games_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/"
        + f"soccer_{sport_id}/team_schedule/"
        + f"{team_id}_team_schedule.csv",
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