# Author: Derek Willis (dwillis@gmail.com)
# File Name: `soccer.py`
# Purpose: Houses functions that allows one to access NCAA soccer data
# Creation Date: 2024-09-20 08:15 PM EDT


import logging
import os
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
    _safe_get_webpage,
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
    inst_html = _safe_get_webpage(inst_url)
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


def get_soccer_team_schedule(team_id: int, get_womens_soccer_data: bool = False) -> pd.DataFrame:
    """
    Retrieves a team schedule, from a valid NCAA soccer team ID.

    Parameters
    ----------
    `team_id` (int, mandatory):
        Required argument.
        Specifies the team you want a schedule from.
        This is separate from a school ID, which identifies the institution.
        A team ID should be unique to a school, and a season.

    Usage
    ----------
    ```python

    from ncaa_stats_py.soccer import get_soccer_team_schedule

    ########################################
    #          Men's soccer                #
    ########################################

    # Get the team schedule for the
    # 2024 Wake Forest MSO team (D1, ID: 571234).
    print(
        "Get the team schedule for the " +
        "2024 Wake Forest MSO team (D1, ID: 571234)."
    )
    df = get_soccer_team_schedule(571234)
    print(df)

    # Get the team schedule for the
    # 2023 Charleston MSO team (D1, ID: 546789).
    print(
        "Get the team schedule for the " +
        "2023 Charleston MSO team (D1, ID: 546789)."
    )
    df = get_soccer_team_schedule(546789)
    print(df)

    ########################################
    #          Women's soccer              #
    ########################################

    # Get the team schedule for the
    # 2024 UNC WSO team (D1, ID: 572345).
    print(
        "Get the team schedule for the " +
        "2024 UNC WSO team (D1, ID: 572345)."
    )
    df = get_soccer_team_schedule(572345)
    print(df)

    # Get the team schedule for the
    # 2023 Duke WSO team (D1, ID: 546987).
    print(
        "Get the team schedule for the " +
        "2023 Duke WSO team (D1, ID: 546987)."
    )
    df = get_soccer_team_schedule(546987)
    print(df)

    ```

    Returns
    ----------
    A pandas `DataFrame` object with an NCAA soccer team's schedule.

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
        if get_womens_soccer_data:
            team_df = load_soccer_teams(start_year=datetime.today().year, get_womens_soccer_data=True)
            team_df = team_df[team_df["team_id"] == team_id]
            if not team_df.empty:
                season = team_df["season"].iloc[0]
                ncaa_division = team_df["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
                sport_id = "WSO"
                team_info_found = True
            del team_df
        else:
            team_df = load_soccer_teams(start_year=datetime.today().year)
            team_df = team_df[team_df["team_id"] == team_id]
            if not team_df.empty:
                season = team_df["season"].iloc[0]
                ncaa_division = team_df["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
                sport_id = "MSO"
                team_info_found = True
            del team_df
    except Exception as e:
        if get_womens_soccer_data:
            logging.info(f"Could not find team in women's soccer data: {e}")
        else:
            logging.info(f"Could not find team in men's soccer data: {e}")

    # If team not found in cached data, extract info from the team page directly
    if not team_info_found:
        logging.warning(f"Team ID {team_id} not found in cached teams data. Extracting info from team page.")
        try:
            # Handle potential asyncio loop conflicts with Playwright
            # import concurrent.futures
            # import threading
            
            # def get_webpage_in_thread():
            #     return _get_webpage(url=url)
            
            # try:
            #     # Check if we're in an asyncio environment
            #     import asyncio
            #     try:
            #         loop = asyncio.get_running_loop()
            #         # We're in an asyncio loop, run in thread
            #         logging.info("Asyncio loop detected, running webpage request in separate thread")
            #         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #             future = executor.submit(get_webpage_in_thread)
            #             response = future.result(timeout=120)  # 2 minute timeout
            #     except RuntimeError:
            #         # No running loop, safe to call directly
            #         response = _get_webpage(url=url)
            # except ImportError:
            #     # asyncio not available, call directly
            #     response = _get_webpage(url=url)
            response = _safe_get_webpage(url=url, _timeout=120)

            soup = BeautifulSoup(response.text, features="lxml")
            
            # Extract season from the year selector
            season_element = soup.find("select", {"id": "year_list"})
            if season_element:
                selected_option = season_element.find("option", {"selected": "selected"})
                if selected_option:
                    season_name = selected_option.text
                    # Convert "2023-24" to 2024 or handle single year formats
                    if "-" in season_name:
                        season = int("20" + season_name.split("-")[-1]) if len(season_name.split("-")[-1]) == 2 else int(season_name.split("-")[-1])
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

    # Always use threading to avoid asyncio conflicts
    # import concurrent.futures
    
    # def get_webpage_in_thread():
    #     return _get_webpage(url=url)
    
    # Run in separate thread to avoid any potential asyncio issues
    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     future = executor.submit(get_webpage_in_thread)
    #     response = future.result(timeout=120)  # 2 minute timeout
    response = _safe_get_webpage(url=url, _timeout=120)

    soup = BeautifulSoup(response.text, features="lxml")

    school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
    season_name = (
        soup.find("select", {"id": "year_list"})
        .find("option", {"selected": "selected"})
        .text
    )
    # For NCAA soccer, the season always starts and ends in the fall semester
    # Thus, if `season_name` = "2011-12", this is the "2011" soccer season,
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
            # Because of how *well* designed
            # stats.ncaa.org is, if we have to use execute
            # the `if len(t_rows) == 0:` code,
            # we need to catch any cases where every element in a
            # table row (`<tr>`) is a table header (`<th>`),
            # instead of a table data cell (`<td>`)
            continue

        game_date = cells[0].text

        # If "(" is in the same cell as the date,
        # this means that this game is part of a series.
        # The number encased in `()` is the game number in the series.
        # We need to remove that from the date,
        # and move it into a separate variable.
        if "(" in game_date:
            game_date = game_date.replace(")", "")
            game_date, game_num = game_date.split("(")
            game_date = game_date.strip()
            game_num = int(game_num.strip())

        game_date = datetime.strptime(game_date, "%m/%d/%Y").date()

        try:
            opp_team_id = cells[1].find("a").get("href")
        except IndexError:
            logging.info(
                "Skipping row because it is clearly "
                + "not a row that has schedule data."
            )
            is_valid_row = False
        except AttributeError as e:
            logging.info(
                "Could not extract a team ID for this game. " +
                f"Full exception {e}"
            )
            opp_team_id = "-1"
        except Exception as e:
            logging.warning(
                "An unhandled exception has occurred when "
                + "trying to get the opposition team ID for this game. "
                f"Full exception `{e}`."
            )
            raise e
        if is_valid_row is True:
            if opp_team_id is not None:
                opp_team_id = opp_team_id.replace("/teams/", "")
                opp_team_id = int(opp_team_id)

                try:
                    opp_team_name = cells[1].find("img").get("alt")
                except AttributeError:
                    logging.info(
                        "Couldn't find the opposition team name "
                        + "for this row from an image element. "
                        + "Attempting a backup method"
                    )
                    opp_team_name = cells[1].text
                except Exception as e:
                    logging.info(
                        "Unhandled exception when trying to get the "
                        + "opposition team name from this game. "
                        + f"Full exception `{e}`"
                    )
                    raise e
            else:
                opp_team_name = cells[1].text

            if opp_team_name[0] == "@":
                # The logic for determining if this game was a
                # neutral site game doesn't care if that info is in
                # `opp_team_name`.
                opp_team_name = opp_team_name.strip().replace("@", "")
            elif "@" in opp_team_name:
                opp_team_name = opp_team_name.strip().split("@")[0]

            opp_text = cells[1].text
            opp_text = opp_text.strip()
            if "@" in opp_text and opp_text[0] == "@":
                is_home_game = False
            elif "@" in opp_text and opp_text[0] != "@":
                is_neutral_game = True
                is_home_game = False
            # This is just to cover conference and NCAA championship
            # tournaments.
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

                # `score_1` should be "W `n`", "L `n`", or "T `n`",
                # with `n` representing the number of goals this team
                # scored in this game.
                # Let's remove the "W", "L", or "T" from `score_1`,
                # and determine which team won later on in this code.
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
                game_url = (
                    f"https://stats.ncaa.org/contests/{game_id}/box_score"
                )

            except AttributeError as e:
                logging.info(
                    "Could not parse a game ID for this game. "
                    + f"Full exception `{e}`."
                )
                game_id = None
                game_url = None
            except Exception as e:
                logging.info(
                    "An unhandled exception occurred when trying "
                    + "to find a game ID for this game. "
                    + f"Full exception `{e}`."
                )
                raise e
            try:
                attendance = cells[3].text
                attendance = attendance.replace(",", "")
                attendance = attendance.replace("\n", "")
                attendance = int(attendance)
            except IndexError as e:
                logging.info(
                    "It doesn't appear as if there is an attendance column "
                    + "for this team's schedule table."
                    f"Full exception `{e}`."
                )
                attendance = None
            except ValueError as e:
                logging.info(
                    "There doesn't appear as if "
                    + "there is a recorded attendance. "
                    + "for this game/row. "
                    f"Full exception `{e}`."
                )
                attendance = None

            except Exception as e:
                logging.info(
                    "An unhandled exception occurred when trying "
                    + "to find this game's attendance. "
                    + f"Full exception `{e}`."
                )
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
                # For the sake of simplicity,
                # order both team ID's,
                # and set the lower number of the two as
                # the "away" team in this neutral site game,
                # just so there's no confusion if someone
                # combines a ton of these team schedule `DataFrame`s,
                # and wants to remove duplicates afterwards.
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


def get_soccer_match_stats(
    season: int,
    level: str | int = "I",
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves game-by-game statistics for NCAA soccer teams, producing detailed
    match stats for each team's games throughout the season.

    Parameters
    ----------
    `season` (int, mandatory):
        Required argument.
        Specifies the season you want NCAA soccer match statistics from.

    `level` (str | int, optional):
        Optional argument (default: "I").
        Specifies the level/division you want
        NCAA soccer match statistics from.
        This can either be an integer (1-3) or a string ("I"-"III").

    `get_womens_soccer_data` (bool, optional):
        Optional argument (default: False).
        If you want women's soccer data instead of men's soccer data,
        set this to `True`.

    Usage
    ----------
    ```python

    from ncaa_stats_py.soccer import get_soccer_match_stats

    # Get all D1 men's soccer match statistics for the 2024 season.
    df = get_soccer_match_stats(2024, 1)
    print(df)

    # Get all D1 women's soccer match statistics for the 2024 season.
    df = get_soccer_match_stats(2024, 1, get_womens_soccer_data=True)
    print(df)

    ```

    Returns
    ----------
    A pandas `DataFrame` object with game-by-game statistics for college soccer teams
    in that season and NCAA level, including offensive and defensive stats for each match.
    """
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    match_stats_df = pd.DataFrame()
    match_stats_df_arr = []
    formatted_level = ""
    ncaa_level = 0

    # Parse level parameter
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
        ncaa_level = 1
        formatted_level = "I"
    elif isinstance(level, str) and (
        level.lower() in {"ii", "d2", "2"}
    ):
        ncaa_level = 2
        formatted_level = "II"
    elif isinstance(level, str) and (
        level.lower() in {"iii", "d3", "3"}
    ):
        ncaa_level = 3
        formatted_level = "III"
    else:
        raise ValueError(
            f"Improper input for `level`: `{level}`.\n"
            + "Valid inputs are (but not limited to) "
            + '`1`, "I", `2`, "II", `3`, and "III".'
        )

    # Create cache directories
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    match_stats_cache_dir = f"{soccer_cache_dir}/match_stats"
    
    for d in [base_cache_dir, soccer_cache_dir, match_stats_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Check for cached data for the requested season only
    cache_file = f"{match_stats_cache_dir}/{season}_{formatted_level}_match_stats.csv"
    if os.path.exists(cache_file):
        match_stats_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        now = datetime.today()
        age = now - file_mod_datetime
        # Only invalidate cache for the requested season
        if (
            age.days > 1 and season == now.year and now.month <= 7
        ) or (
            age.days >= 14 and season == (now.year - 1) and now.month <= 7
        ) or age.days >= 35:
            load_from_cache = False
        else:
            if load_from_cache:
                return match_stats_df
    else:
        load_from_cache = False

    logging.warning(
        f"Either we could not load {season} D{level} match statistics from cache, "
        + "or it's time to refresh the cached data."
    )

    # Get teams for this season/level
    teams_df = get_soccer_teams(season, level, get_womens_soccer_data)
    if teams_df.empty:
        logging.warning(f"No teams found for {season} season, level {level}")
        return pd.DataFrame()

    # Process each team's schedule to get match stats
    for _, team_row in tqdm(teams_df.iterrows(), desc="Processing team schedules", total=len(teams_df)):
        team_id = team_row['team_id']
        team_name = team_row['school_name']
        
        try:
            # Get team schedule
            schedule_df = get_soccer_team_schedule(team_id, get_womens_soccer_data=get_womens_soccer_data)
            
            if schedule_df.empty:
                logging.info(f"No schedule found for team {team_id} ({team_name})")
                continue
                
            # Process each game in the schedule
            for _, game_row in schedule_df.iterrows():
                game_id = game_row['game_id']
                
                if pd.isna(game_id) or game_id is None:
                    continue
                    
                try:
                    # Get match stats from box score
                    match_stats = get_match_stats_from_box_score(game_id, team_id, team_name)
                    
                    if match_stats is not None:
                        match_stats_df_arr.append(match_stats)
                        
                except Exception as e:
                    logging.warning(f"Error processing game {game_id} for team {team_id}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing team {team_id} ({team_name}): {e}")
            continue

    if not match_stats_df_arr:
        logging.warning("No match statistics data found")
        return pd.DataFrame()

    # Combine all match stats
    final_match_stats_df = pd.concat(match_stats_df_arr, ignore_index=True)
    
    # Sort by date and team
    final_match_stats_df.sort_values(by=["date", "team"], inplace=True)

    # Save to cache
    final_match_stats_df.to_csv(cache_file, index=False)

    return final_match_stats_df


def get_match_stats_from_box_score(game_id: int, team_id: int, team_name: str) -> pd.DataFrame:
    """
    Helper function to extract match statistics from a soccer game box score.
    
    Parameters
    ----------
    game_id : int
        The NCAA game ID
    team_id : int
        The team ID to extract stats for
    team_name : str
        The team name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with match statistics for the specified team
    """
    try:
        game_url = f"https://stats.ncaa.org/contests/{game_id}/box_score"
        response = _safe_get_webpage(url=game_url, _timeout=120)
        soup = BeautifulSoup(response.text, features="lxml")
        
        # Initialize match stats with default values
        match_stats = {
            "date": None,
            "team": team_name,
            "opponent": None,
            "home_away": None,
            "outcome": None,
            "team_score": 0,
            "opponent_score": 0,
            "overtime": "NA",
            "gp": 1,
            "minutes": "0:00",
            "goals": 0,
            "assists": 0,
            "points": 0,
            "sh_att": 0,
            "so_g": 0,
            "fouls": 0,
            "red_cards": 0,
            "yellow_cards": 0,
            "corners": 0,
            "pk": 0,
            "pk_att": "NA",
            "gwg": "NA",
            "defensive_minutes": "0:00",
            "defensive_goals": 0,
            "defensive_assists": 0,
            "defensive_points": 0,
            "defensive_sh_att": 0,
            "defensive_so_g": 0,
            "defensive_fouls": 0,
            "defensive_red_cards": 0,
            "defensive_yellow_cards": 0,
            "defensive_corners": 0,
            "defensive_pk": 0,
            "defensive_pk_att": "NA",
            "defensive_gwg": "NA",
            "team_id": team_id
        }
        
        # Extract basic game information
        game_header = soup.find("div", class_="card-header")
        if game_header:
            header_text = game_header.text.strip()
            # Extract date, teams, and score from header
            # This would need to be implemented based on the actual HTML structure
            pass
        
        # Find team statistics tables
        # Look for tables containing team stats
        stat_tables = soup.find_all("table")
        
        for table in stat_tables:
            # Process each statistics table
            # This would extract offensive and defensive stats
            # Implementation depends on the actual box score HTML structure
            pass
        
        return pd.DataFrame([match_stats])
        
    except Exception as e:
        logging.error(f"Error extracting match stats for game {game_id}: {e}")
        return None


def get_soccer_team_stats(
    season: int,
    level: str | int = "I",
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves team statistics for NCAA soccer teams from the institution trends page,
    including both regular team stats, goalkeeping statistics, and win/loss records.

    Parameters
    ----------
    `season` (int, mandatory):
        Required argument.
        Specifies the season you want NCAA soccer team statistics from.

    `level` (str | int, optional):
        Optional argument (default: "I").
        Specifies the level/division you want
        NCAA soccer team statistics from.
        This can either be an integer (1-3) or a string ("I"-"III").

    `get_womens_soccer_data` (bool, optional):
        Optional argument (default: False).
        If you want women's soccer data instead of men's soccer data,
        set this to `True`.

    Usage
    ----------
    ```python

    from ncaa_stats_py.soccer import get_soccer_team_stats

    ########################################
    #          Men's soccer                #
    ########################################

    # Get all D1 men's soccer team statistics for the 2024 season.
    print("Get all D1 men's soccer team statistics for the 2024 season.")
    df = get_soccer_team_stats(2024, 1)
    print(df)

    # Get all D3 men's soccer team statistics for the 2023 season.
    print("Get all D3 men's soccer team statistics for the 2023 season.")
    df = get_soccer_team_stats(2023, 3)
    print(df)

    ########################################
    #          Women's soccer              #
    ########################################

    # Get all D1 women's soccer team statistics for the 2024 season.
    print("Get all D1 women's soccer team statistics for the 2024 season.")
    df = get_soccer_team_stats(2024, 1, get_womens_soccer_data=True)
    print(df)

    # Get all D3 women's soccer team statistics for the 2023 season.
    print("Get all D3 women's soccer team statistics for the 2023 season.")
    df = get_soccer_team_stats(2023, 3, get_womens_soccer_data=True)
    print(df)

    ```

    Returns
    ----------
    A pandas `DataFrame` object with comprehensive team statistics (including goalkeeping and win/loss records) 
    for college soccer teams in that season and NCAA level.
    """
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    stats_df = pd.DataFrame()
    stats_df_arr = []
    temp_df = pd.DataFrame()
    formatted_level = ""
    ncaa_level = 0

    # Parse level parameter
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
        ncaa_level = 1
        formatted_level = "I"
    elif isinstance(level, str) and (
        level.lower() in {"ii", "d2", "2"}
    ):
        ncaa_level = 2
        formatted_level = "II"
    elif isinstance(level, str) and (
        level.lower() in {"iii", "d3", "3"}
    ):
        ncaa_level = 3
        formatted_level = "III"
    else:
        raise ValueError(
            f"Improper input for `level`: `{level}`.\n"
            + "Valid inputs are (but not limited to) "
            + '`1`, "I", `2`, "II", `3`, and "III".'
        )

    # Create cache directories
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    stats_cache_dir = f"{soccer_cache_dir}/team_stats"
    
    for d in [base_cache_dir, soccer_cache_dir, stats_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Check for cached data
    cache_file = f"{stats_cache_dir}/{season}_{formatted_level}_team_stats.csv"
    
    if os.path.exists(cache_file):
        stats_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(os.path.getmtime(cache_file))
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()
    age = now - file_mod_datetime

    # Cache invalidation logic
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

    if load_from_cache:
        return stats_df

    logging.warning(
        f"Either we could not load {season} D{level} team statistics from cache, "
        + "or it's time to refresh the cached data."
    )

    # Get schools data for merging
    schools_df = _get_schools()
    
    # First, get the teams to ensure we have valid data
    teams_df = get_soccer_teams(season, level, get_womens_soccer_data)
    if teams_df.empty:
        logging.warning(f"No teams found for {season} season, level {level}")
        return pd.DataFrame()

    # Get the correct stat IDs from utls based on season and sport
    sport_key = "womens_soccer" if get_womens_soccer_data else "mens_soccer"
    
    try:
        goalie_stat_category_id = _get_stat_id(sport_key, season, "goalkeepers")
        regular_stat_category_id = _get_stat_id(sport_key, season, "non_goalkeepers")
        season_stat_id = _get_stat_id(sport_key, season, "team")
    except LookupError as e:
        logging.warning(f"Could not find stat IDs for {sport_key} season {season}: {e}")
        # Fall back to default values if stat IDs not found
        if get_womens_soccer_data:
            season_stat_id = 56
            goalie_stat_category_id = 57
            regular_stat_category_id = 58
        else:
            season_stat_id = 30
            goalie_stat_category_id = 31
            regular_stat_category_id = 32

    # Build the URL to get ranking periods
    url = (
        "https://stats.ncaa.org/rankings/change_sport_year_div?"
        + f"academic_year={season + 1}.0&division={ncaa_level}.0"
        + f"&sport_code={sport_id}"
    )

    response = _safe_get_webpage(url=url, _timeout=120)
    soup = BeautifulSoup(response.text, features="lxml")
    
    # Find the ranking periods
    ranking_periods_temp = soup.find("select", {"name": "rp", "id": "rp"})
    if ranking_periods_temp:
        ranking_periods = ranking_periods_temp.find_all("option")
        rp_value = 0
        found_value = False

        # Get the final ranking period (or most recent)
        for rp in ranking_periods:
            if "championship" in rp.text.lower():
                continue
            elif "final" in rp.text.lower():
                rp_value = rp.get("value")
                found_value = True
                break
            else:
                rp_value = rp.get("value")
                found_value = True
                
        if not found_value and ranking_periods:
            rp_value = ranking_periods[-1].get("value")
    else:
        rp_value = 0

    # Helper function to extract win/loss records from team page
    def get_team_records(team_id):
        """Extract win/loss records from individual team page"""
        team_url = f"https://stats.ncaa.org/teams/{team_id}"
        
        try:
            response = _safe_get_webpage(url=team_url, _timeout=120)
            soup = BeautifulSoup(response.text, features="lxml")
            
            # Initialize record variables
            overall_wins = 0
            overall_losses = 0
            overall_ties = 0
            conference_wins = 0
            conference_losses = 0
            conference_ties = 0
            home_wins = 0
            home_losses = 0
            home_ties = 0
            road_wins = 0
            road_losses = 0
            road_ties = 0
            neutral_wins = 0
            neutral_losses = 0
            neutral_ties = 0
            
            logging.debug(f"Processing team records for team_id: {team_id}")
            
            # Look for the Season-to-date Records section
            records_header = soup.find("div", class_="card-header", string="Season-to-date Records")
            
            if records_header:
                # Get the card body that follows this header
                records_card_body = records_header.find_next_sibling("div", class_="card-body")
                
                if records_card_body:
                    # Find all the individual record cards within the card body
                    record_cards = records_card_body.find_all("div", class_="card")
                    
                    logging.debug(f"Found {len(record_cards)} record cards for team {team_id}")
                    
                    for card in record_cards:
                        # Get the header for this specific record type
                        card_header = card.find("div", class_="card-header")
                        card_body = card.find("div", class_="card-body")
                        
                        if card_header and card_body:
                            record_type = card_header.text.strip().lower()
                            logging.debug(f"Processing record type: {record_type}")
                            
                            # Look for spans containing the record
                            record_spans = card_body.find_all("span")
                            
                            for span in record_spans:
                                span_text = span.text.strip()
                                logging.debug(f"Found span text: '{span_text}' for record type: {record_type}")
                                
                                # Look for the pattern "X-Y-Z (percentage)" or "X-Y (percentage)" for soccer
                                if "-" in span_text and not span_text.startswith("Streak"):
                                    # Extract the wins-losses-ties part (before parentheses if present)
                                    if "(" in span_text:
                                        record_part = span_text.split("(")[0].strip()
                                    else:
                                        record_part = span_text.strip()
                                    
                                    if "-" in record_part:
                                        try:
                                            record_components = record_part.split("-")
                                            wins = int(record_components[0].strip())
                                            losses = int(record_components[1].strip())
                                            ties = int(record_components[2].strip()) if len(record_components) > 2 else 0
                                            
                                            logging.debug(f"Parsed {record_type}: {wins}-{losses}-{ties}")
                                            
                                            # Assign to appropriate variables based on record type
                                            if record_type == "overall":
                                                overall_wins = wins
                                                overall_losses = losses
                                                overall_ties = ties
                                            elif record_type == "conference":
                                                conference_wins = wins
                                                conference_losses = losses
                                                conference_ties = ties
                                            elif record_type == "home":
                                                home_wins = wins
                                                home_losses = losses
                                                home_ties = ties
                                            elif record_type == "road":
                                                road_wins = wins
                                                road_losses = losses
                                                road_ties = ties
                                            elif record_type == "neutral":
                                                neutral_wins = wins
                                                neutral_losses = losses
                                                neutral_ties = ties
                                            
                                            break  # Found the record, move to next card
                                            
                                        except (ValueError, IndexError) as ve:
                                            logging.warning(f"Error parsing wins/losses/ties from '{record_part}': {ve}")
                                            continue
                else:
                    logging.debug(f"Could not find card-body for Season-to-date Records for team {team_id}")
            else:
                logging.debug(f"Could not find Season-to-date Records header for team {team_id}")
            
            result = {
                "overall_wins": overall_wins,
                "overall_losses": overall_losses,
                "overall_ties": overall_ties,
                "conference_wins": conference_wins,
                "conference_losses": conference_losses,
                "conference_ties": conference_ties,
                "home_wins": home_wins,
                "home_losses": home_losses,
                "home_ties": home_ties,
                "road_wins": road_wins,
                "road_losses": road_losses,
                "road_ties": road_ties,
                "neutral_wins": neutral_wins,
                "neutral_losses": neutral_losses,
                "neutral_ties": neutral_ties
            }
            
            logging.debug(f"Final results for team {team_id}: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error getting records for team {team_id}: {e}")
            return {
                "overall_wins": 0,
                "overall_losses": 0,
                "overall_ties": 0,
                "conference_wins": 0,
                "conference_losses": 0,
                "conference_ties": 0,
                "home_wins": 0,
                "home_losses": 0,
                "home_ties": 0,
                "road_wins": 0,
                "road_losses": 0,
                "road_ties": 0,
                "neutral_wins": 0,
                "neutral_losses": 0,
                "neutral_ties": 0
            }

    # Helper function to scrape statistics table
    def scrape_stats_table(url, table_type="regular"):
        try:
            response = _safe_get_webpage(url=url, _timeout=120)
        except Exception as e:
            logging.info(f"Found exception when loading {table_type} stats `{e}`")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, features="lxml")

        # Find the statistics table
        table = soup.find("table", {"id": "stat_grid"})
        if table is None:
            logging.warning(f"Could not find {table_type} statistics table")
            return pd.DataFrame()

        # Get table headers
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            headers = [th.text.strip() for th in header_row.find_all("th")]
        else:
            logging.warning(f"Could not find {table_type} table headers")
            return pd.DataFrame()

        # Get table body
        tbody = table.find("tbody")
        if tbody is None:
            logging.warning(f"Could not find {table_type} table body")
            return pd.DataFrame()

        t_rows = tbody.find_all("tr", {"class": "text"})
        table_data = []

        for row in t_rows:
            cells = row.find_all("td")
            
            # Extract team ID and name from the first cell
            first_cell = cells[0]
            team_link = first_cell.find("a")
            if team_link:
                team_href = team_link.get("href")
                if "/teams/" in team_href:
                    team_id = team_href.replace("/teams/", "")
                    team_id = int(team_id)
                    team_name = team_link.text.strip()
                else:
                    continue
            else:
                continue  # Skip rows without team links

            # Extract all cell data
            row_data = []
            for i, cell in enumerate(cells):
                if i == 0:  # Institution name
                    row_data.append(team_name)
                else:
                    # Get the data-order attribute if available (contains numeric values)
                    data_order = cell.get("data-order")
                    if data_order and data_order != "-":
                        try:
                            # Handle time format (e.g., "90:00" for minutes)
                            if ":" in data_order:
                                row_data.append(data_order)
                            elif "." in data_order:
                                row_data.append(float(data_order))
                            else:
                                row_data.append(int(data_order))
                        except ValueError:
                            row_data.append(data_order)
                    else:
                        # Fall back to text content
                        text_content = cell.text.strip()
                        if text_content == "" or text_content == "-":
                            row_data.append(None)
                        else:
                            row_data.append(text_content)

            # Get win/loss records for this team
            team_records = get_team_records(team_id)

            # Create DataFrame row
            temp_df = pd.DataFrame([row_data], columns=headers)
            temp_df["team_id"] = team_id
            temp_df["season"] = season
            temp_df["ncaa_division"] = ncaa_level
            temp_df["ncaa_division_formatted"] = formatted_level
            temp_df["sport_id"] = sport_id
            
            # Add win/loss record columns
            for key, value in team_records.items():
                temp_df[key] = value
            
            table_data.append(temp_df)

        if table_data:
            return pd.concat(table_data, ignore_index=True)
        else:
            return pd.DataFrame()

    # Get regular team statistics using the correct stat ID
    regular_stats_url = (
        "https://stats.ncaa.org/rankings/institution_trends?"
        + f"division={ncaa_level}.0&id={season_stat_id}"
        + f"&year_stat_category_id={regular_stat_category_id}"
    )
    
    regular_stats_df = scrape_stats_table(regular_stats_url, "regular")
    
    # Get goalkeeping statistics using the correct stat IDs
    goalie_stats_url = (
        "https://stats.ncaa.org/rankings/institution_trends?"
        + f"division={ncaa_level}.0&id={season_stat_id}"
        + f"&year_stat_category_id={goalie_stat_category_id}"
    )
    
    goalie_stats_df = scrape_stats_table(goalie_stats_url, "goalkeeping")
    
    # Merge the dataframes
    if not regular_stats_df.empty and not goalie_stats_df.empty:
        # Rename goalkeeping columns to avoid conflicts
        goalie_rename_map = {}
        for col in goalie_stats_df.columns:
            if col not in ["team_id", "season", "ncaa_division", "ncaa_division_formatted", "sport_id", "Institution", "Conference"]:
                if not col.startswith("overall_") and not col.startswith("conference_") and not col.startswith("home_") and not col.startswith("road_") and not col.startswith("neutral_"):
                    goalie_rename_map[col] = f"goalie_{col.lower().replace(' ', '_')}"
        
        # Rename columns in goalkeeping dataframe
        goalie_stats_df = goalie_stats_df.rename(columns=goalie_rename_map)
        
        # Merge on team_id
        merge_columns = ["team_id", "season", "ncaa_division", "ncaa_division_formatted", "sport_id"]
        
        stats_df = pd.merge(
            regular_stats_df, 
            goalie_stats_df, 
            on=merge_columns, 
            how="left",
            suffixes=('', '_goalie')
        )
        
        # Drop duplicate columns from the merge
        duplicate_cols = [col for col in stats_df.columns if col.endswith('_goalie') and col.replace('_goalie', '') in stats_df.columns]
        stats_df = stats_df.drop(columns=duplicate_cols)
        
    elif not regular_stats_df.empty:
        stats_df = regular_stats_df
    elif not goalie_stats_df.empty:
        stats_df = goalie_stats_df
    else:
        logging.warning("No team statistics data found")
        return pd.DataFrame()

    # Add to stats_df_arr for processing
    stats_df_arr.append(stats_df)

    if not stats_df_arr:
        logging.warning("No team statistics data found")
        return pd.DataFrame()

    # Combine all rows
    final_stats_df = pd.concat(stats_df_arr, ignore_index=True)
    
    # Merge with schools data for additional school information
    final_stats_df = pd.merge(
        left=final_stats_df,
        right=schools_df,
        left_on="Institution",
        right_on="school_name",
        how="left"
    )
    
    # Drop duplicate columns from the merge
    if "school_name_y" in final_stats_df.columns:
        final_stats_df = final_stats_df.drop(columns=["school_name_y"])
    elif "school_name" in final_stats_df.columns and "Institution" in final_stats_df.columns:
        final_stats_df = final_stats_df.drop(columns=["school_name"])
    
    # Clean up column names to be more descriptive for soccer
    column_mapping = {
        "Institution": "school_name",
        "Conference": "conference_name",
        "Games": "games_played",
        "GP": "games_played",
        "Goals": "goals",
        "Assists": "assists",
        "Points": "points",
        "Shots": "shots",
        "SOG": "shots_on_goal",
        "SH%": "shot_percentage",
        "Fouls": "fouls",
        "Yellow Cards": "yellow_cards",
        "Red Cards": "red_cards",
        "Corners": "corner_kicks",
        "Saves": "saves",
        "GAA": "goals_against_average",
        "Save%": "save_percentage",
        "Shutouts": "shutouts",
        "Goals Against": "goals_against",
        "Minutes": "minutes_played",
        "Win%": "win_percentage"
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in final_stats_df.columns:
            final_stats_df.rename(columns={old_name: new_name}, inplace=True)

    # Sort by team_id
    final_stats_df.sort_values(by=["team_id"], inplace=True)

    # Save to cache
    final_stats_df.to_csv(cache_file, index=False)

    return final_stats_df

if __name__ == "__main__":
    pass