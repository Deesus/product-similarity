<template>
    <div class="d-flex flex-column justify-center align-center fill-height">
        <h1 v-if="error.statusCode === 404" class="font-weight-light">
            {{ pageNotFound }}
        </h1>
        <h1 v-else>
            {{ otherError }}
        </h1>
        <div class="py-10">
            <v-btn text color="primary" @click="reloadHomePage">
                Back to Home page
            </v-btn>
        </div>
    </div>
</template>

<script>
export default {
    name: 'EmptyLayout',
    layout: 'empty',
    props: {
        error: {
            type: Object,
            default: null
        }
    },
    data() {
        return {
            pageNotFound: '404 Not Found',
            otherError: 'An error occurred'
        }
    },
    head() {
        const title = this.error.statusCode === 404 ? this.pageNotFound : this.otherError
        return {
            title
        }
    },
    methods: {
        reloadHomePage() {
            this.$router
                .push('/')
                // Redirecting to home doesn't reset the error page; we have to refresh to do that:
                .then(window.location.reload())
        }
    }
}
</script>
