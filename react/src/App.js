import './App.css';
import React, { Component } from 'react';

import axios from 'axios';

import Questionnaire from './components/Questionnaire';
import Resources from './components/Resources';
import NavBar from './components/NavBar';
import GetPosts from './components/GetPosts';

import $ from 'jquery';

const max_posts = 200;

class App extends Component {
    constructor(props) {
        super(props)

        this.state = {
            'currPage': "GetPosts",

            'username': "",
            'writings': [],
            'predictions': Array(21).fill(undefined),
            'selections': Array(21).fill(undefined),

            'alertColor': 'warning',
            'alertText': undefined,
            'alertExpiration': 0
        }
    }

    sleep = (ms) => {
        return new Promise(resolve => setTimeout(resolve, ms))
    }

    showAlert = (text, level="warning", duration=4000) => {
        if (level === "warning") {
            console.warn(text)
        } else {
            console.info(text)
        }
        

        $("div[class*=alert]").each(function() {
            $(this).removeClass("hidden")
        });
        
        this.setState({
            alertColor: level,
            alertText: text,
            alertExpiration: Date.now() + duration
        })

        this.sleep(duration+100).then(() => {
            if (Date.now() >= this.state.alertExpiration) {
                $("div[class*=alert]").each(function() {
                    $(this).addClass("hidden")
                });
            }
        })
    }
    
    // NAVBAR

    handleNavClick = (e) => {
        this.setState({
            currPage: e.target.name
        })
        $("a[class*=nav-link]").each(function() {
            if ($(this).attr("name") === e.target.name) {
              $(this).addClass("active")
            } else {
              $(this).removeClass("active")
            }
        });
    }

    // GETPOSTS

    handleUserPredict = () => {
        if (this.state.username === undefined || this.state.username === '') {
            this.showAlert(`Cannot predict results for user. Fetch their posts first.`)
            return false;
        }
        if (this.state.writings === undefined || this.state.writings.length === 0) {
            this.showAlert(`Cannot predict results for user with no posts`)
            return false;
        }

        this.showAlert(`Predicting responses for user '${this.state.username}'...`, 'primary', 300000)
        let data = {subject: this.state.username, writings: this.state.writings}
        let url = "/post"
        if (process.env.REACT_APP_ENV === "prod") {
            // url = "https://us-central1-reddit-bdi.cloudfunctions.net/post"
            url = "https://backend-dot-reddit-bdi.uc.r.appspot.com/post"
        }
        console.log("Predict from " + url)

        axios.post(url, data)
            .then(response => {
                let predictions = response.data.responses
                this.setState({
                    predictions: predictions
                })

                if (this.state.predictions === undefined || this.state.predictions.length === 0) {
                    this.showAlert(`Server unable to predict responses.`)
                    return false;
                }

                this.showAlert(`Predicted responses for user '${this.state.username}'`, 'success')

                this.setState({
                    currPage: "Questionnaire",
                    selections: this.state.predictions
                })

                $("a[class*=nav-link]").each(function() {
                    if ($(this).attr("name") === "Questionnaire") {
                    $(this).addClass("active")
                    } else {
                    $(this).removeClass("active")
                    }
                });

                return true
            })
            .catch(this.handleAxiosError);
    }

    handleUserUpdate = (username) => {
        this.setState({
            "username": username
        })
    }

    getPosts = (username, before=0, depth=0, type='submission') => {
        let before_param = before !== 0 ? `&before=${before}` : ""
        if (depth===Math.ceil(max_posts / 100)) {
            if (type === 'submission') {
                return this.sleep(Math.min(depth*150,1000)).then(r => 
                    this.getPosts(username, 0, 0, 'comment')
                )
            } else {
                return this.sleep(0).then(r => {return []})
            }
        }
        return axios.get(`https://api.pushshift.io/reddit/search/${type}/?sort=desc&size=100&author=${username}${before_param}`)
        .then(response => {
            let posts = response.data.data
            if (posts.length < 100) {
                if (type === 'submission') {
                    return this.sleep(Math.min(depth*150,1000)).then(r => 
                        this.getPosts(username, 0, 0, 'comment')
                            .then(nextPosts => posts.concat(nextPosts))
                    )
                } else {
                    return posts
                }
            } else {
                let nextBefore = posts[posts.length-1]['created_utc']
                return this.sleep(Math.min(depth*150,1000)).then(r => 
                    this.getPosts(username, nextBefore, depth+1, type)
                        .then(nextPosts => posts.concat(nextPosts))
                )
            }
        }).catch((e) => {
            this.handleAxiosError(e);
            return [];
        })
    };

    // NOTE: Using username instead of this.state.username may cause problems if the user presses submit before it's done.
    handleSearch = (username) => {
        if (username === undefined || username === '') {
            this.showAlert(`Cannot get reddit posts for user '${username}'`)
            return false;
        }
        
        this.setState({
            'writings': [],
            'username': username,
            'predictions': []
        })
        this.showAlert(`Fetching reddit posts for user '${username}'...`, 'primary', 60000)
        this.getPosts(username).then(posts => {
            posts.sort(function(first, second) {
                return second.created_utc - first.created_utc;
            });
            posts = posts.slice(0, max_posts)
            let post_dicts = posts.map((post) => {
                // TODO: Do pre-processing to remove links & extraneous punctuation?
                let text = ""
                let info = "reddit submission"
                let title = ""
                if ('body' in post) {
                    text = post.body
                    info = "reddit comment"
                    title = ""
                } else if ('selftext' in post && 'title' in post) {
                    text = post.selftext
                    info = "reddit post"
                    title = post.title
                }
                let date = new Date(post.created_utc * 1000);
        
                return {'TITLE': title, 'DATE': date, 'INFO': info, 'TEXT': text};
            });
            let writings = this.state.writings.concat(post_dicts)
            this.setState({
                writings: writings
            })
            this.showAlert(`Fetched reddit posts for user '${username}'`, 'success')
            return true
        })
        .catch(this.handleAxiosError);
    }

    handleAxiosError = (error) => {
        if (error.response) {
            // Request made and server responded
            console.error(error.response)
            this.showAlert("Error in server response: " + error.response.status + " - " + error.response.statusText, 'danger');
        } else if (error.request) {
            // The request was made but no response was received
            console.error(error.request)
            this.showAlert("No reponse recieved from server.", 'danger');
        } else {
            // Something happened in setting up the request that triggered an Error
            console.error(error.message)
            this.showAlert("Error: " + error.message, 'danger');
        }
        return false
    }

    // QUESTIONNAIRE

    clearSelections = () => {
        this.setState({
            selections: Array(21).fill(undefined)
        })
    }

    quickFillSelections = () => {
        this.setState({
            selections: Array(21).fill("0")
        })
    }

    handleChangeSelections = (e) => {
        let arr = this.state.selections
        let idx = parseInt(e.target.name)
        if (e.target.checked) {
            arr[idx] = e.target.value
        } else {
            arr[idx] = undefined
        }
        this.setState({
            selections: arr
        });
    }

    shareBdiResults = (username) => {
        this.setState({
            username: "",
            'writings': [],
            'predictions': Array(21).fill(undefined)
        })

        if (username === undefined || username === '') {
            this.showAlert(`Cannot share results for user '${username}'`)
            return false;
        }
        if (this.state.selections === undefined || this.state.selections.length !== 21 || this.state.selections.filter(value => value !== undefined).length !== 21) {
            this.showAlert(`Cannot share results for incomplete user questionnaire`)
            return false;
        }

        this.showAlert(`Sharing responses from user '${username}'...`, "primary")

        let data = {subject: username, responses: this.state.selections}
        let url = "/post"
        if (process.env.REACT_APP_ENV === "prod") {
            // url = "https://us-central1-reddit-bdi.cloudfunctions.net/post"
            url = "https://backend-dot-reddit-bdi.uc.r.appspot.com/post"
        }

        console.log("Connecting to " + url)

        axios.post(url, data)
            .then(response => {
                this.showAlert(response.data, "success")
                return true
            })
            .catch(this.handleAxiosError);
    }

    // TODO: Pass predictions to questionnaire somehow
    render() {

        return (
            <div className="App">
                <script src="/docs/5.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-JEW9xMcG8R+pH31jmWH6WWP0WintQrMb4s7ZOdauHnUtxwoG2vI5DkLtS3qm9Ekf" crossOrigin="anonymous"></script>
                <script src="https://kit.fontawesome.com/0fd4c3f81b.js" crossorigin="anonymous"></script>
                <NavBar handleClick={this.handleNavClick} />

                {/* <main className="container"> */}
                    <div className="bg-light p-5 mx-5 mt-3 mb-4 rounded">
                        <Resources currPage={this.state.currPage} />
                        <Questionnaire currPage={this.state.currPage} selections={this.state.selections} predictions={this.state.predictions} handleChange={this.handleChangeSelections} quickFill={this.quickFillSelections} clearSelections={this.clearSelections} handleShareResults={this.shareBdiResults}/>
                        <GetPosts currPage={this.state.currPage} handlePredict={this.handleUserPredict} handleSearch={this.handleSearch} handleUserUpdate={this.handleUserUpdate} writings={this.state.writings} username={this.state.username} />
                    </div>
                {/* </main> */}

                {this.state.alertText === undefined ? <></> :
                    <div className="d-flex p-2 justify-content-end mr-3 fixed-bottom">
                        <div className={`alert alert-${this.state.alertColor}`} style={{width: "auto"}} role="alert">
                            {!this.state.alertText.includes("...") ? <></> :
                                <div className={`spinner-border spinner-border-sm mr-3 text-${this.state.alertColor}`} role="status">
                                </div>
                            }
                            {this.state.alertText}
                        </div>
                    </div>
                }
            </div>
        );
    }
}

export default App;
