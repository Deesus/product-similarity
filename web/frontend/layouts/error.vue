<template>
    <v-app light class="text-center">
        <h1 v-if="error.statusCode === 404" class="font-weight-light">
            {{ pageNotFound }}
        </h1>
        <h1 v-else>
            {{ otherError }}
        </h1>
        <NuxtLink to="/" class="pt-5">
            <v-btn text color="primary">
                Back to Home page
            </v-btn>
        </NuxtLink>
    </v-app>
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
    }
}
</script>
