$(document).ready(function() {


                    var cluster_id = $('#cluster_id').val();
                    alert(cluster_id);

                    var c_container = document.getElementById("cluster_container");
                    tweet = document.createElement("div");
                    tweet.setAttribute("id", "tweet");
                    c_container.appendChild(tweet);

                        twttr.widgets.createTweet(cluster_id, tweet, {
                        conversation : "all",    // or all
                        cards        : "hidden",  // or hidden
                        linkColor    : "#cc0000", // default is blue
                        theme        : "light"   // or dark
                    })
                    .then (function (el) {
                        console.log("Tweet Displayed");
                    });
                } );