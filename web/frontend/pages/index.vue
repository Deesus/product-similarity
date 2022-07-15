<template>
    <div>
        <!-- ----- Header: ----- -->
        <v-row justify="center" align="center" class="py-6">
            <v-col cols="12" class="d-flex justify-center align-center">
                <h1 class="font-weight-light">
                    Product <span class="text--secondary">Image Search</span>
                </h1>
            </v-col>
        </v-row>

        <!-- ----- Upload image dropzone: ----- -->
        <v-row justify="center" align="center">
            <v-col cols="12" sm="8" md="6">
                <FileUpload
                    :thumbnail="thumbnail"
                    :is-loading="isLoading"
                    @get-products="getProducts"
                    @select-img="setThumbnail"
                    @error="showErrorAlert"
                />
            </v-col>
        </v-row>

        <!-- ----- Error/Warning alert: ----- -->
        <v-row justify="center">
            <v-col cols="12" sm="8" md="6">
                <v-alert
                    v-show="uploadErrorText"
                    dense
                    outlined
                    type="error"
                >
                    {{ uploadErrorText }}
                </v-alert>
            </v-col>
        </v-row>

        <!-- ----- Display results: ----- -->
        <!-- We need to use `v-show` instead of `v-if` because we are watching the `filPaths` prop; since the component
        is not rendered initially, no props exist and therefore there isn't a prop change the first time: -->
        <v-row v-show="filePaths.length" class="mt-15">
            <v-col cols="12">
                <ListResults
                    :file-paths="filePaths"
                    @select-img="setThumbnail"
                    @loading="setLoading"
                    @error="showErrorAlert"
                />
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
            filePaths: [],
            thumbnail: null,
            isLoading: false,
            uploadErrorText: ''
        }
    },
    // Set the page title <https://nuxtjs.org/docs/concepts/views/#pages>:
    head() {
        return {
            title: 'Product Similarity | Deepankara Reddy'
        }
    },
    methods: {
        getProducts({ 'file-paths': filePaths }) {
            this.filePaths = filePaths
        },
        /**
         * Set thumbnail image.
         *
         * @param file {String | Blob}: The thumbnail image; can be a file path or blob. If string type, then arg is a
         *      file path (url); if it is blob, the file is the uploaded image.
         */
        setThumbnail(file) {
            if(!file) {
                return
            }

            // See <https://jsfiddle.net/jykmapb8> -- most tutorials use the same method for creating an upload preview:
            if(file instanceof Blob) {
                const reader = new FileReader()
                reader.onload = (event) => {
                    this.thumbnail = event.target.result
                }
                reader.readAsDataURL(file)
            } else {
                this.thumbnail = file
            }
        },
        /**
         * Set data loading state handler.
         *
         * This simply means showing the progress bar on components while server processes/fetches data.
         *
         * @param isLoading {Boolean}: Emitted value specifying if data is being loaded.
         */
        setLoading(isLoading) {
            this.isLoading = isLoading
        },
        showErrorAlert(message) {
            this.uploadErrorText = message
        }
    }
}
</script>
