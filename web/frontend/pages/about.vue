<template>
    <v-row justify="center" align="center">
        <v-col cols="12" sm="8" md="6">
            <v-card>
                <v-card-title class="headline">
                    Search For Similar Products by Image
                </v-card-title>
                <v-card-text>
                    <div class="text-xs-right">
                        <em><small>&mdash; {{ message }}</small></em>
                    </div>
                    <hr class="my-3">
                    <a
                        href="https://nuxtjs.org/"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Nuxt Documentation
                    </a>
                    <br>
                    <a
                        href="https://github.com/nuxt/nuxt.js"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Nuxt GitHub
                    </a>
                </v-card-text>
                <v-card-actions>
                    <v-spacer />
                    <v-btn
                        color="primary"
                        nuxt
                        @click="getMessage"
                    >
                        Continue
                    </v-btn>
                </v-card-actions>
            </v-card>
        </v-col>
    </v-row>
</template>

<script>
import axios from 'axios'

export default{
    name: 'AboutPage',
    data() {
        return{
            message: ''
        }
    },
    methods: {
        getMessage() {
            const path = 'http://localhost:5000/ping'
            axios.get(path)
                .then((response) => {
                    this.message = response.data
                })
                .catch((error) => {
                    // eslint-disable-next-line no-console
                    console.error(error)
                })
        }
    }
}
</script>
