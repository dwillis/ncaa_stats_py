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

def get_soccer_team_schedule(
    team_id: int, 
    season: int, 
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves a team schedule, from a valid NCAA soccer team ID.
    
    Parameters
    ----------
    team_id : int
        The NCAA team ID
    season : int
        The season year (e.g., 2025 for Fall 2025 season)
    get_womens_soccer_data : bool, optional
        Whether to get women's soccer data (default: False for men's)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the team's schedule
    """
    
    # Set sport_id based on parameter
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    
    schools_df = _get_schools()
    games_df = pd.DataFrame()
    games_df_arr = []
    temp_df = pd.DataFrame()
    load_from_cache = True

    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    # Try to get team info from teams cache first
    team_info_found = False
    ncaa_division = 1
    ncaa_division_formatted = "I"
    school_name = None
    
    try:
        # Load teams for the specified sport and find the team
        teams_df = get_soccer_teams(season, "I", get_womens_soccer_data=get_womens_soccer_data)
        if not teams_df.empty:
            team_row = teams_df[teams_df["team_id"] == team_id]
            if not team_row.empty:
                ncaa_division = team_row["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                school_name = team_row["school_name"].iloc[0]
                team_info_found = True
                logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
        
        # If not found in Division I, try other divisions
        if not team_info_found:
            for div in ["II", "III"]:
                teams_df = get_soccer_teams(season, div, get_womens_soccer_data=get_womens_soccer_data)
                if not teams_df.empty:
                    team_row = teams_df[teams_df["team_id"] == team_id]
                    if not team_row.empty:
                        ncaa_division = team_row["ncaa_division"].iloc[0]
                        ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                        school_name = team_row["school_name"].iloc[0]
                        team_info_found = True
                        logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
                        break
                        
    except Exception as e:
        logging.warning(f"Could not find team in teams cache: {e}")

    # If team not found in cache, extract info from the team page directly
    if not team_info_found:
        logging.warning(f"Team ID {team_id} not found in teams cache. Extracting info from team page.")
        try:
            url = f"https://stats.ncaa.org/teams/{team_id}"
            response = _get_webpage(url=url)
            soup = BeautifulSoup(response.text, features="lxml")
            
            # Extract school name from the page
            try:
                school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
            except:
                # Fallback method
                try:
                    school_name = soup.find("title").text.split(" - ")[0]
                except:
                    school_name = f"Team {team_id}"
            
            logging.info(f"Extracted school name: {school_name}")
            
        except Exception as e:
            logging.error(f"Could not extract team info from page: {e}")
            school_name = f"Team {team_id}"

    # Get stat sequence for the sport and season
    try:
        sport_key = "womens_soccer" if get_womens_soccer_data else "mens_soccer"
        stat_sequence = _get_stat_id(sport_key, season, "team")
        logging.info(f"Found stat ID {stat_sequence} for {sport_key} season {season}")
    except LookupError:
        logging.warning(f"Could not find team stat ID for {sport_key} season {season}")
        # Use fallback values
        stat_sequence = 56 if get_womens_soccer_data else 30

    # Ensure cache directories exist
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    schedule_cache_dir = f"{soccer_cache_dir}/team_schedule"
    
    for d in [base_cache_dir, soccer_cache_dir, schedule_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    cache_file = f"{schedule_cache_dir}/{team_id}_team_schedule.csv"

    # Check if we should load from cache
    if os.path.exists(cache_file):
        games_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(os.path.getmtime(cache_file))
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()
    age = now - file_mod_datetime
    
    # Determine if cache is stale
    if age.days > 1 and season >= now.year and now.month <= 7:
        load_from_cache = False

    if load_from_cache:
        return games_df

    # Scrape the schedule from the team page
    url = f"https://stats.ncaa.org/teams/{team_id}"
    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    # Get school name if we don't have it yet
    if not school_name:
        try:
            school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
        except:
            school_name = f"Team {team_id}"

    # Get season name
    try:
        season_name = (
            soup.find("select", {"id": "year_list"})
            .find("option", {"selected": "selected"})
            .text
        )
    except:
        season_name = f"{season}-{str(season+1)[-2:]}"

    # Find the schedule table
    soup_sections = soup.find_all("div", {"class": "col p-0"})
    
    table_data = None
    for s in soup_sections:
        try:
            temp_name = s.find("div", {"class": "card-header"})
            temp_name = temp_name.text
        except Exception as e:
            logging.warning(f"Could not parse card header. Full exception `{e}`. Attempting alternate method.")
            try:
                temp_name = s.find("tr", {"class": "heading"}).find("td").text
            except:
                continue

        if "schedule" in temp_name.lower():
            table_data = s.find("table")
            break

    if table_data is None:
        logging.warning("Could not find schedule table on page")
        return pd.DataFrame()

    # Parse schedule rows
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

        # Parse game date
        game_date = cells[0].text.strip()
        if not game_date or game_date.lower() in ['date', 'game']:
            continue

        # Handle game series notation
        if "(" in game_date:
            game_date = game_date.replace(")", "")
            game_date, game_num = game_date.split("(")
            game_date = game_date.strip()
            game_num = int(game_num.strip())

        try:
            game_date = datetime.strptime(game_date, "%m/%d/%Y").date()
        except ValueError:
            continue  # Skip invalid date rows

        # Parse opponent
        try:
            opp_team_link = cells[1].find("a")
            if opp_team_link:
                opp_team_id = opp_team_link.get("href")
                if opp_team_id:
                    opp_team_id = opp_team_id.replace("/teams/", "")
                    opp_team_id = int(opp_team_id)
                else:
                    opp_team_id = None
            else:
                opp_team_id = None
        except (IndexError, AttributeError, ValueError):
            logging.info("Skipping row because it doesn't contain valid team data.")
            continue
        except Exception as e:
            logging.warning(f"Unhandled exception when parsing opponent team ID: {e}")
            opp_team_id = None

        # Parse opponent name
        try:
            opp_team_name = cells[1].find("img").get("alt")
        except AttributeError:
            opp_team_name = cells[1].text.strip()
        except Exception as e:
            logging.warning(f"Error parsing opponent name: {e}")
            continue

        # Determine home/away/neutral
        opp_text = cells[1].text.strip()
        if "@" in opp_text and opp_text.startswith("@"):
            is_home_game = False
            opp_team_name = opp_team_name.replace("@", "").strip()
        elif "@" in opp_text and not opp_text.startswith("@"):
            is_neutral_game = True
            is_home_game = False
            opp_team_name = opp_text.split("@")[0].strip()
        elif any(keyword in opp_text.lower() for keyword in ["championship", "ncaa", "tournament"]):
            is_neutral_game = True
            is_home_game = False

        # Parse score
        score = cells[2].text.strip()
        if len(score) == 0 or any(word in score.lower() for word in ["canceled", "ppd", "postponed"]):
            score_1 = None
            score_2 = None
        else:
            try:
                # Handle overtime notation
                score = score.replace("(-3 OT)", "")
                
                # Remove win/loss indicators
                if any(x in score for x in ["W", "L", "T"]):
                    parts = score.split(" ")
                    if len(parts) >= 2:
                        score = parts[1]

                score_1, score_2 = score.split("-")
                
                # Handle overtime periods
                if "(" in score_2:
                    score_2 = score_2.replace(")", "")
                    score_2, ot_text = score_2.split("(")
                    ot_periods = int(re.findall(r'\d+', ot_text)[0]) if re.findall(r'\d+', ot_text) else 0

                score_1 = int(score_1.strip())
                score_2 = int(score_2.strip())
            except (ValueError, IndexError):
                score_1 = None
                score_2 = None

        # Parse game ID and URL
        try:
            game_link = cells[2].find("a")
            if game_link:
                game_href = game_link.get("href")
                game_id = int(re.findall(r'/contests/(\d+)', game_href)[0])
                game_url = f"https://stats.ncaa.org/contests/{game_id}/box_score"
            else:
                game_id = None
                game_url = None
        except (AttributeError, IndexError, ValueError):
            game_id = None
            game_url = None

        # Parse attendance
        try:
            if len(cells) > 3:
                attendance = cells[3].text.strip().replace(",", "").replace("\n", "")
                attendance = int(attendance) if attendance.isdigit() else None
            else:
                attendance = None
        except (ValueError, IndexError):
            attendance = None

        # Create game record
        if is_home_game:
            temp_df = pd.DataFrame({
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
                "attendance": attendance,
            }, index=[0])
        elif is_neutral_game:
            # Order team IDs for consistent neutral game representation
            if opp_team_id and team_id:
                t_ids = [opp_team_id, team_id]
                t_ids.sort()
                
                if t_ids[0] == team_id:
                    temp_df = pd.DataFrame({
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
                        "attendance": attendance,
                    }, index=[0])
                else:
                    temp_df = pd.DataFrame({
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
                        "attendance": attendance,
                    }, index=[0])
            else:
                # Fallback if opp_team_id is None
                temp_df = pd.DataFrame({
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
                    "attendance": attendance,
                }, index=[0])
        else:  # Away game
            temp_df = pd.DataFrame({
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
                "attendance": attendance,
            }, index=[0])

        games_df_arr.append(temp_df)

    if not games_df_arr:
        return pd.DataFrame()

    # Combine all games
    games_df = pd.concat(games_df_arr, ignore_index=True)

    # Merge with school information
    temp_df = schools_df.rename(columns={
        "school_name": "home_team_name",
        "school_id": "home_school_id"
    })
    games_df = games_df.merge(right=temp_df, on="home_team_name", how="left")

    temp_df = schools_df.rename(columns={
        "school_name": "away_team_name", 
        "school_id": "away_school_id"
    })
    games_df = games_df.merge(right=temp_df, on="away_team_name", how="left")
    
    # Add division information
    games_df["ncaa_division"] = ncaa_division
    games_df["ncaa_division_formatted"] = ncaa_division_formatted

    # Save to cache
    games_df.to_csv(cache_file, index=False)

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