import json
import requests
import os
import argparse
import time
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from threading import Thread, Lock
import re
import glob
import shutil

# paths used to save the datasets obtained
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_RAW_CORPUS = os.path.join(CURRENT_PATH, 'raw')
PATH_INTERIM_CORPUS = os.path.join(CURRENT_PATH, 'interim')

# token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'

# values used to filter the final dataset
MIN_NUM_WRITINGS = 30
MIN_AVG_WORDS = 15

# information related to the requests to reddit
NUMBER_LIMIT = 100
USER_AGENT = 'chrome_win7_64:erisk_bot:v1 (erisk_2021)'
TIMEOUT_LIMIT = 5

# regexes
USER_NAME_REGEX = re.compile(r'\bu/[a-zA-Z\-0-9_]+')
BOT_REGEX = re.compile(r"(\bi(\)? \^{0,2}\(?a|')m\)? \^{0,2}\(?a\)? \^{0,2}\(?(ro)?bot\)?\b)|(this is a bot\b)",
                       flags=re.IGNORECASE)

# list of excluded users, posts and subreddits
EXCLUDED_USERS = ['[deleted]', 'AutoModerator', 'LoansBot', 'GoodBot_BadBot', 'B0tRank']
EXCLUDED_POSTS = ['[deleted]', '[removed]', 'removed by moderator']
EXCLUDED_SUBREDDITS = ['copypasta']


def request_get(url, headers, params):
    """Send a GET request and retries up to five times in case it fails.

    Parameters
    ----------
    url : str
        URL to send GET request.
    headers : dict
        HTTP headers information to send with the request. We only used
        'User-agent'.
    params : dict
        Parameters send with the request string. We used 'limit', 'after',
        'include_over_18'.

    Returns
    -------
    r : requests.Response
        Request response.
    status : int
        Status code of the request.

    Notes
    -----
    In case the request fails, the function waits for 5 seconds and retry
    the same request again. After 5 attempts, return the last failed
    result.
    If the request takes longer than the value defined in the global
    variable TIMEOUT_LIMIT, the request returns the status code 408.

    Examples
    --------
    >>> h = {'User-agent': 'erisk_2021'}
    >>> param = {'limit': 100}
    >>> u = 'reddit.com/top.json'
    >>> request_get(url=u, headers=h, params=param)
    (<Response [200]>, 200)
    """
    num_tries = 0
    status = 503
    r = None
    while status != 200 and num_tries < 5:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_LIMIT)
            status = r.status_code
        except requests.exceptions.Timeout:
            print(f'The GET request took longer than {TIMEOUT_LIMIT} seconds.')
            status = 408
        except requests.exceptions.ConnectionError:
            print('The maximum number of retries for the URL has been exceeded.')
            status = 429
        num_tries = num_tries + 1
        if status != 200:
            print('The GET request failed, trying again.')
            time.sleep(5)
    return r, status


def process_comments(replies, ids, dicts, link, current_post_time, subreddit_name):
    """Recursively process comments from post or comment.

    Store the author, time, and content of the comments of a post.

    Parameters
    ----------
    replies : dict
        Dictionary with the replies to the post or comment to process.
    ids : set
        Set of ids of the user of interest.
    dicts : dict
        Dictionary with the information of every user's post.
    link : str
        URL of the current post being processed. In case a comment it is
         being processed it makes references to the post the comment
         belongs to.
    current_post_time : int
        Time of the current post being processed. In case a comment it
        is being processed it makes references to the post the comment
        belongs to.
    subreddit_name : str
        Name of the subreddit of interest.
    """
    for element in replies['data']['children']:
        if element['kind'] == 't1':
            # It is a comment
            current_comment_author = element['data']['author']
            comment_time = element['data']['created_utc']
            comment_content = element['data']['body']
            current_subreddit = element['data']['subreddit']

            if current_comment_author not in EXCLUDED_USERS and comment_content not in EXCLUDED_POSTS and \
                    BOT_REGEX.search(comment_content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
                # If there is a reference to a user, we remove it
                comment_content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=comment_content)

                if subreddit_name == current_subreddit:
                    ids.add(current_comment_author)

                if current_comment_author not in dicts:
                    dicts[current_comment_author] = {}
                if link not in dicts[current_comment_author]:
                    dicts[current_comment_author][link] = {
                        'content': '',
                        'time': current_post_time,
                        'comments': [],
                    }
                dicts[current_comment_author][link]['comments'].append({
                    'content': comment_content,
                    'time': comment_time,
                })

            if element['data']['replies'] != '':
                process_comments(replies=element['data']['replies'], ids=ids, dicts=dicts, link=link,
                                 current_post_time=current_post_time, subreddit_name=subreddit_name)


def process_post(link, ids, posts_already_processed, subreddit_name, lock, output_directory):
    """Process post from Reddit.

    Store the author, time, content and comments of a post.

    Parameters
    ----------
    link : str
        URL of the post to process.
    ids : set
        Set of ids of the user of interest.
    posts_already_processed : set
        Set of posts already processed.
    subreddit_name : str
        Name of the subreddit of interest.
    lock : threading.Lock
        A lock object to ensure correct writing of the shared resources.
    output_directory : str
        Directory path to save the auxiliary dictionary with all the
        information from the post.
    """
    len_suffix = len('.json')
    current_post_url = link[:-len_suffix]
    if current_post_url in posts_already_processed:
        return

    r, request_status_code = request_get(link, headers={'User-agent': USER_AGENT}, params={'limit': NUMBER_LIMIT})

    if request_status_code != 200:
        return

    json_r = r.json()

    # Get information from post
    json_post = json_r[0]
    assert len(json_post['data']['children']) == 1

    post_information = json_post['data']['children'][0]

    post_author = post_information['data']['author']
    current_subreddit = post_information['data']['subreddit']
    time_current_post = post_information['data']['created_utc']
    title_post = post_information['data']['title']
    body_post = post_information['data']['selftext']
    content = title_post + ' ' + body_post
    post_identifier = post_information['data']['name']

    aux_dicts = {}
    aux_ids = set()

    if post_author not in EXCLUDED_USERS and title_post not in EXCLUDED_POSTS and body_post not in EXCLUDED_POSTS and \
            BOT_REGEX.search(content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
        # If there is a reference to a user, we remove it
        content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=content)

        if subreddit_name == current_subreddit:
            aux_ids.add(post_author)

        aux_dicts[post_author] = {}
        aux_dicts[post_author][link] = {
            'content': content,
            'time': time_current_post,
            'comments': [],
        }

    # Get information from the comments
    json_comments = json_r[1]
    for element in json_comments['data']['children']:
        if element['kind'] == 't1':
            # It is a comment
            comment_author = element['data']['author']
            current_subreddit = element['data']['subreddit']
            comment_time = element['data']['created_utc']
            comment_content = element['data']['body']

            if comment_author not in EXCLUDED_USERS and comment_content not in EXCLUDED_POSTS and\
                    BOT_REGEX.search(comment_content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
                # If there is a reference to a user, we remove it
                comment_content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=comment_content)

                if subreddit_name == current_subreddit:
                    aux_ids.add(comment_author)
                if comment_author not in aux_dicts:
                    aux_dicts[comment_author] = {}

                if link not in aux_dicts[comment_author]:
                    aux_dicts[comment_author][link] = {
                        'content': '',
                        'time': time_current_post,
                        'comments': [],
                    }
                aux_dicts[comment_author][link]['comments'].append({
                    'content': comment_content,
                    'time': comment_time,
                })

            if element['data']['replies'] != '':
                process_comments(replies=element['data']['replies'], ids=aux_ids, dicts=aux_dicts, link=link,
                                 current_post_time=time_current_post, subreddit_name=subreddit_name)
        else:
            continue

    # We write the dictionary with all the information from the post in a file
    aux_dicts_path = os.path.join(output_directory, f'{post_identifier}.json')
    with open(aux_dicts_path, "w") as fp:
        json.dump(fp=fp, obj=aux_dicts, indent='\t')

    # We write the data in the shared data structures
    with lock:
        ids.union(aux_ids)
        posts_already_processed.add(current_post_url)


def sort_comments(post_comments):
    """Sort comments based on the time they were posted.

        Parameters
        ----------
        post_comments : list of dict
            List of dictionaries with comments of a user in a post.
            Each dictionary has two keys: time and content.

        Returns
        -------
        comments_sorted : list of str
            List of str with comments of a user in a post sorted.
        number_comments : int
            Number of comments of the user in a particular post.
        """
    number_comments = len(post_comments)
    sorted_comments_with_times = sorted(post_comments, key=lambda comentario: comentario['time'])
    comments_sorted = [c['content'] for c in sorted_comments_with_times]

    return comments_sorted, number_comments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to build a corpus using reddit posts.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['t1', 't2'])
    parser.add_argument("mode", help="Script mode", choices=['append', 'overwrite', 'keep'])
    args = parser.parse_args()

    main_subreddit = 'selfharm' if args.corpus == 't2' else 'problemgambling'

    partial_raw_output_path = os.path.join(PATH_RAW_CORPUS, args.corpus)
    post_list_path = os.path.join(partial_raw_output_path, f'reddit-{args.corpus}-post-list.json')
    id_users_path = os.path.join(partial_raw_output_path, f'reddit-{args.corpus}-id-users.json')

    os.makedirs(partial_raw_output_path, exist_ok=True)

    id_users_subreddit = set()
    posts_list = set()

    if os.path.isfile(post_list_path):
        if args.mode == 'overwrite':
            print('Overwriting the previously generated reddit corpus')
            os.remove(post_list_path)
            os.remove(id_users_path)
            # We remove all saved posts too
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_new'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_top'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_users'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'random_users'))
        elif args.mode == 'keep':
            print('The reddit corpus is already created. Aborting script')
            sys.exit()
        elif args.mode == 'append':
            print('The reddit corpus is already created. Loading old data to append new posts')
            with open(post_list_path, "r") as f:
                posts_list = json.load(fp=f)
                posts_list = set(posts_list)
            with open(id_users_path, "r") as f:
                id_users_subreddit = json.load(fp=f)
                id_users_subreddit = set(id_users_subreddit)

    # We look for the most recent posts on the main subreddit
    get_url = f'https://www.reddit.com/r/{main_subreddit}/new.json'
    id_last_post = None
    num_children = NUMBER_LIMIT
    print('-·'*40)
    print(f'Getting posts from the subreddit {main_subreddit}')

    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_new')
    os.makedirs(dict_output_directory, exist_ok=True)

    threads = []
    write_lock = Lock()
    while num_children == NUMBER_LIMIT:
        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post})

        if status_code != 200:
            num_children = 0
            continue

        json_response = get_response.json()
        num_children = len(json_response["data"]["children"])
        print(f'Getting next set of posts from the subreddit {main_subreddit}')

        for p in json_response["data"]["children"]:
            id_users_subreddit.add(p['data']['author'])
            post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'

            thread = Thread(target=process_post,
                            kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                    'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                    'lock': write_lock, 'output_directory': dict_output_directory})
            threads.append(thread)
            thread.start()

        if num_children > 0:
            id_last_post = json_response["data"]["children"][-1]["data"]['name']

    for t in threads:
        t.join()

    # We look for the top posts of the subreddit
    get_url = f'https://www.reddit.com/r/{main_subreddit}/top/.json?sort=top&t=all'
    id_last_post = None
    num_children = NUMBER_LIMIT
    print('-·' * 40)
    print(f'Getting top posts from the subreddit {main_subreddit}')

    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_top')
    os.makedirs(dict_output_directory, exist_ok=True)

    threads = []
    write_lock = Lock()
    while num_children == NUMBER_LIMIT:
        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post})

        if status_code != 200:
            num_children = 0
            continue

        json_response = get_response.json()
        num_children = len(json_response["data"]["children"])
        print(f'Getting next set of top posts from the subreddit {main_subreddit}')

        for p in json_response["data"]["children"]:
            id_users_subreddit.add(p['data']['author'])
            post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'

            thread = Thread(target=process_post,
                            kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                    'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                    'lock': write_lock, 'output_directory': dict_output_directory})
            threads.append(thread)
            thread.start()

        if num_children > 0:
            id_last_post = json_response["data"]["children"][-1]["data"]['name']

    for t in threads:
        t.join()

    # Now, we have to go through the history of posts of the users in the stored list
    print('-·' * 40)
    print('Obtaining posts from users of interest')

    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_users')
    os.makedirs(dict_output_directory, exist_ok=True)

    for user_name in id_users_subreddit.copy():
        print(f'Collecting information from the posts of the user {user_name}')
        get_url = f'https://www.reddit.com/search.json?q=author:{user_name}'
        id_last_post = None
        num_children = NUMBER_LIMIT

        threads = []
        while num_children == NUMBER_LIMIT:
            get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                    params={'limit': NUMBER_LIMIT, 'after': id_last_post,
                                                            'include_over_18': 'on'})

            if status_code != 200:
                num_children = 0
                continue

            json_response = get_response.json()
            num_children = len(json_response["data"]["children"])
            print(f'Getting next set of posts from user {user_name}')

            for p in json_response["data"]["children"]:
                post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'

                thread = Thread(target=process_post,
                                kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                        'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                        'lock': write_lock, 'output_directory': dict_output_directory})
                threads.append(thread)
                thread.start()

            if num_children > 0:
                id_last_post = json_response["data"]["children"][-1]["data"]['name']

        for t in threads:
            t.join()

    # We get posts from randoms users to populate negative cases
    print('-·' * 40)
    general_subreddits = ['sports',
                          'jokes',
                          'gaming',
                          'politics',
                          'news',
                          'LifeProTips',
                          ]
    general_users = set()

    for sub in general_subreddits:
        # Get the last 100 posts from each subreddit.
        # We limit the retrieval to 100 post because otherwise the corpus skews to the
        # negative class.
        get_url = f'https://www.reddit.com/r/{sub}/new.json'
        id_last_post = None
        print(f'Getting randoms users from the subreddit {sub}')

        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post})

        if status_code != 200:
            continue

        json_response = get_response.json()
        print(f'Getting next set of users from the subreddit {sub}')

        for p in json_response["data"]["children"]:
            current_author = p['data']['author']
            if current_author not in EXCLUDED_USERS and current_author not in id_users_subreddit:
                general_users.add(current_author)

    # Now we have to go through the history of random users' posts
    print('-·' * 40)
    print('Getting posts from random users')

    dict_output_directory = os.path.join(partial_raw_output_path, 'random_users')
    os.makedirs(dict_output_directory, exist_ok=True)

    for user_name in general_users:
        print(f'Collecting information from the posts of the user {user_name}')
        get_url = f'https://www.reddit.com/search.json?q=author:{user_name}'
        id_last_post = None

        threads = []
        # Get the last 100 posts from each user.
        # We limit the retrieval to 100 post because otherwise the corpus skews to the
        # negative class.
        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post,
                                                        'include_over_18': 'on'})

        if status_code != 200:
            continue

        json_response = get_response.json()
        print(f'Getting next set of posts from user {user_name}')

        for p in json_response["data"]["children"]:
            post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'

            thread = Thread(target=process_post,
                            kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                    'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                    'lock': write_lock, 'output_directory': dict_output_directory})
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

    print(f"Saving the list of posts processed in {post_list_path}")
    with open(post_list_path, "w") as f:
        json.dump(fp=f, obj=list(posts_list), indent='\t')

    print(f"Saving the list of positive users in {id_users_path}")
    with open(id_users_path, "w") as f:
        json.dump(fp=f, obj=list(id_users_subreddit), indent='\t')

    # We generate the corpus in txt format already processed
    print('-·' * 40)
    print('Processing corpus: concatenate posts, sort them chronologically '
          'and filter users with few or banned posts')

    user_name_list = []
    post_url_list = []
    post_body_list = []
    post_subreddit_list = []
    post_time_list = []
    num_writings_list = []
    num_words_list = []

    len_prefix = len('https://www.reddit.com/r/')

    for file_path in glob.iglob(f'{partial_raw_output_path}/*/*.json'):
        with open(file_path, "r") as f:
            user_dicts = json.load(fp=f)
        for user_name, writings in user_dicts.items():
            for post_url, post_content in writings.items():
                idx_suffix = post_url[len_prefix:].find('/') + len_prefix

                sorted_comments, num_comments = sort_comments(post_content['comments'])

                title = post_content['content']
                join_sorted_comments = END_OF_POST_TOKEN.join(sorted_comments)
                if join_sorted_comments != '' and title != '':
                    post_body = title + END_OF_POST_TOKEN + join_sorted_comments
                else:
                    post_body = title + join_sorted_comments

                # Remove repeated white spaces, new lines and tabs.
                post_body = ' '.join(post_body.split())
                post_body = post_body + END_OF_POST_TOKEN

                post_subreddit = post_url[len_prefix:idx_suffix]
                post_time = int(post_content['time'])
                num_writings = num_comments if post_content['content'] == '' else num_comments + 1
                num_words = len(post_body.replace(END_OF_POST_TOKEN, ' ').split())

                user_name_list.append(user_name)
                post_url_list.append(post_url)
                post_body_list.append(post_body)
                post_subreddit_list.append(post_subreddit)
                post_time_list.append(post_time)
                num_writings_list.append(num_writings)
                num_words_list.append(num_words)

    df_user_postings = pd.DataFrame({
        "user_name": user_name_list,
        "post_url": post_url_list,
        "post_body": post_body_list,
        "post_subreddit": post_subreddit_list,
        "post_time": post_time_list,
        "num_writings": num_writings_list,
        "num_words": num_words_list,
    })

    df_user_postings = df_user_postings.astype({
        "user_name": 'object',
        "post_url": 'object',
        "post_body": 'object',
        "post_subreddit": 'object',
        "post_time": 'int32',
        "num_writings": 'int32',
        "num_words": 'int32',
    })

    ag_sum = df_user_postings.groupby('user_name').sum()
    selected_users = ag_sum[ag_sum.num_writings > MIN_NUM_WRITINGS].index.to_list()
    filtered_df_user_postings = df_user_postings[df_user_postings.user_name.isin(selected_users)]
    print(f'Number of users with more than {MIN_NUM_WRITINGS} posts: {len(selected_users)}')

    ag_sum = filtered_df_user_postings.groupby('user_name').sum()
    avg_num_words = ag_sum.num_words / ag_sum.num_writings
    avg_num_words = avg_num_words.astype(int)
    selected_users = avg_num_words[avg_num_words > MIN_AVG_WORDS].index.to_list()
    filtered_df_user_postings = filtered_df_user_postings[filtered_df_user_postings.user_name.isin(selected_users)]
    print(f'Number of users that additionally have {MIN_AVG_WORDS} words '
          f'on average per post: {len(selected_users)}')

    user_list = filtered_df_user_postings.user_name.unique()

    partial_interim_output_path = os.path.join(PATH_INTERIM_CORPUS, args.corpus)
    corpus_file_path = os.path.join(partial_interim_output_path, f'reddit-{args.corpus}-raw.txt')

    os.makedirs(os.path.dirname(corpus_file_path), exist_ok=True)

    labels = []
    documents = []
    for user in user_list:
        user_df = filtered_df_user_postings[filtered_df_user_postings.user_name == user]
        label = 'positive' if (user_df.post_subreddit == main_subreddit).any() else 'negative'
        labels.append(label)
        document = user_df.sort_values(by='post_time').post_body.sum()[:-len(END_OF_POST_TOKEN)]
        documents.append(document)

        with open(corpus_file_path, 'a', encoding='utf-8') as f:
            f.write(label + '\t' + document + '\n')

    # We separate the corpus in training and testing
    print('-·' * 40)
    print('Saving the training and test corpus already separated')
    documents_train, documents_test, labels_train, labels_test = train_test_split(documents, labels, test_size=0.5,
                                                                                  stratify=labels, random_state=30)

    train_file_path = os.path.join(partial_interim_output_path, f'{args.corpus}-train-raw.txt')
    for i, document in enumerate(documents_train):
        with open(train_file_path, 'a', encoding='utf-8') as f:
            f.write(labels_train[i] + '\t' + document + '\n')

    test_file_path = os.path.join(partial_interim_output_path, f'{args.corpus}-test-raw.txt')
    for i, document in enumerate(documents_test):
        with open(test_file_path, 'a', encoding='utf-8') as f:
            f.write(labels_test[i] + '\t' + document + '\n')

    print('#' * 50)
    print('End of the script')
