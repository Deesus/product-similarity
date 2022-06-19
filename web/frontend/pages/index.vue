<template>
    <div>
        <!-- ----- Header: ----- -->
        <v-row justify="center" align="center" class="py-6">
            <v-col cols="12" class="d-flex justify-center align-center">
                <v-img class="logo-package" src="/package.svg" alt="logo of package box" />
                <h1 class="font-weight-light ml-5">
                    Product <span class="text--secondary">Image Search</span>
                </h1>
            </v-col>
        </v-row>

        <!-- ----- Upload image dropzone: ----- -->
        <v-row justify="center" align="center">
            <v-col cols="12" sm="8" md="6">
                <FileUpload @get-products="getProducts" @set-is-loading="setIsLoading" />
            </v-col>
        </v-row>

        <!-- ----- Display results: ----- -->
        <v-row class="mt-15">
            <v-col cols="12">
                <ListResults :images="images" :is-loading="isLoading" />
            </v-col>
        </v-row>
    </div>
</template>

<script>
import FileUpload from '~/components/FileUpload'
import ListResults from '~/components/ListResults'

export default {
    name: 'IndexPage',
    components: { FileUpload, ListResults },
    data() {
        return {
            images: [],
            isLoading: false
        }
    },
    methods: {
        getProducts({ 'file-paths': filePaths }) {
            this.images = filePaths
        },
        setIsLoading(val) {
            this.isLoading = val
        }
    }
}
</script>

<style scoped lang="scss">
    .logo-package {
        width: 100px;
        max-width: 100px;
        height: auto;
    }
</style>
